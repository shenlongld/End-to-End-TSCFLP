# 保留分组路径强化学习和原生 RL4CO rollout 基线训练流程。

import copy
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from rl4co.data.dataset import TensorDictDataset
from rl4co.models.rl.reinforce.baselines import RolloutBaseline
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from tensordict import TensorDict

from .common import parse_config, set_seed, validate_device
from .dataset import (
    FixedDatasetGenerator,
    dataset_identity,
    load_checkpoint,
    load_dataset_directory,
    save_checkpoint,
)
from .env import PathTSCFLPEnv, TSCFLPEnv
from .generator import TSCFLPGenerator
from .infer import best_of_k
from .metrics import episode_metrics, inference_metrics
from .model import PathTSCFLPPolicy, TSCFLPPolicy


def validate_data_mode(config):
    if config.data_mode not in ("synthetic", "npz"):
        raise ValueError("data_mode must be synthetic or npz")
    if config.data_mode == "npz":
        if config.dataset_dir is None:
            raise ValueError("npz training requires dataset_dir")
        Path(config.dataset_dir).resolve(strict=True)
    elif config.dataset_dir is not None:
        raise ValueError("dataset_dir is only valid for data_mode=npz")


def check_training_data(config, training, validation, train_size):
    if training.batch_size[0] != train_size or validation.batch_size[0] != config.val_size:
        raise ValueError("Training and validation sizes must match the complete NPZ splits")
    dimensions = tuple(
        training[key].shape[-1]
        for key in ("warehouse_capacity", "station_capacity", "customer_demand")
    )
    if dimensions != (config.n_w, config.n_s, config.n_c):
        raise ValueError("Dataset dimensions do not match the training configuration")


def validate_training_paths(config):
    validate_device(config.device)
    validate_data_mode(config)
    for name in ("resume", "initialize_from"):
        value = getattr(config, name, None)
        if value is not None:
            path = Path(value).resolve(strict=True)
            if path.suffix != ".npz":
                raise ValueError("Checkpoints must be pickle-free NPZ archives")
    if config.resume and getattr(config, "initialize_from", None):
        raise ValueError("resume and initialize_from have different optimizer semantics")
    if not config.resume and (Path(config.output) / "last.npz").exists():
        raise ValueError("Output already has a run; use resume or a new directory")


def grouped_rl_loss(
    reward, log_likelihood, entropy=None, best_weight=0.25, reward_scale=1000.0, entropy_weight=0.0
):
    # 输入形状为 [实例数,K]，每组 K 个独立同分布样本，不强制起点或按排名缩放。
    # 均值项：E[R_i - mean(R_-i)] * grad(log p_i)，对 i 求平均。
    # Max@K 项：E[max(R_all) - max(R_-i)] * grad(log p_i)，对 i 求和。
    # Max@K 项不能再除以 K，否则会改变指定的混合目标。
    # 可选熵项仅用于局部探索正则，不包含在无偏得分函数估计的结论中。
    # 基线和优势均与梯度图分离。
    if reward.ndim != 2 or reward.shape != log_likelihood.shape or reward.shape[1] < 2:
        raise ValueError("Expected matching [B,K] tensors with K >= 2")
    if not 0 <= best_weight <= 1 or not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Invalid best_weight or reward_scale")
    if not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("entropy_weight must be finite and nonnegative")
    r = reward.detach() / reward_scale
    k = r.shape[1]
    mean_baseline = (r.sum(-1, True) - r) / (k - 1)
    mean_advantage = r - mean_baseline
    other = (
        r[:, None, :]
        .expand(-1, k, -1)
        .masked_fill(torch.eye(k, dtype=torch.bool, device=r.device)[None], -torch.inf)
        .amax(-1)
    )
    best_advantage = r.amax(-1, True) - other
    mean_loss = -(mean_advantage * log_likelihood).mean()
    best_loss = -(best_advantage * log_likelihood).sum(-1).mean()
    loss = (1 - best_weight) * mean_loss + best_weight * best_loss
    if entropy is not None:
        loss = loss - entropy_weight * entropy.mean()
    return loss, {
        "mean_advantage": mean_advantage,
        "best_advantage": best_advantage,
        "mean_loss": mean_loss.detach(),
        "best_loss": best_loss.detach(),
    }


@dataclass
class PathTrainConfig:
    epochs: int = 30
    train_instances: int = 256
    group_size: int = 8
    batch_instances: int = 8
    val_size: int = 32
    val_samples: int = 8
    sample_chunk: int = 8
    n_w: int = 3
    n_s: int = 10
    n_c: int = 100
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    dynamic_context: bool = True
    lr: float = 1e-4
    best_weight: float = 0.25
    best_warmup_epochs: int = 2
    entropy_weight: float = 0.002
    reward_scale: float = 1000.0
    penalty: float = 1e5
    seed: int = 45
    threads: int = 1
    device: str = ""
    data_mode: str = "synthetic"
    output: str = "checkpoints/path"
    dataset_dir: str | None = None
    resume: str | None = None


def _cpu_weights(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def train_path(config: PathTrainConfig):
    validate_training_paths(config)
    for key in (
        "epochs",
        "train_instances",
        "batch_instances",
        "val_size",
        "val_samples",
        "sample_chunk",
        "threads",
    ):
        if getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive")
    if config.group_size < 2 or config.best_warmup_epochs < 0:
        raise ValueError("group_size >= 2 and best_warmup_epochs >= 0 required")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    validate_device(config.device)
    torch.set_num_threads(config.threads)
    set_seed(config.seed, device=config.device)
    torch.use_deterministic_algorithms(True)
    generator_config = dict(n_w=config.n_w, n_s=config.n_s, n_c=config.n_c)
    validate_data_mode(config)
    generator = TSCFLPGenerator(**generator_config, seed=config.seed + 10000)
    training_data, dataset_metadata = None, None
    if config.data_mode == "npz":
        splits, dataset_metadata = load_dataset_directory(config.dataset_dir, ("train", "val"))
        training_data, validation = splits["train"], splits["val"]
        check_training_data(config, training_data, validation, config.train_instances)
    else:
        validation = TSCFLPGenerator(**generator_config, seed=config.seed + 20000)(
            [config.val_size]
        )
    env = PathTSCFLPEnv(penalty=config.penalty)
    policy = PathTSCFLPPolicy(
        config.embed_dim,
        config.num_heads,
        config.num_layers,
        dynamic_context=config.dynamic_context,
    ).to(config.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=0.0)
    output = Path(config.output)
    if not config.resume and (output / "last.npz").exists():
        raise ValueError("Output already has a run; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    history, first_epoch, best_key, best_epoch = [], 1, (float("inf"), float("inf")), 0
    best_weights = _cpu_weights(policy)
    initial_metrics = None
    if config.resume:
        saved = load_checkpoint(config.resume)
        if dataset_identity(saved["dataset_metadata"]) != dataset_identity(dataset_metadata):
            raise ValueError("Resume dataset content or path changed")
        if saved.get("architecture") != "coupled_path_v1" or "optimizer_state_dict" not in saved:
            raise ValueError("Resume requires path last.npz with optimizer and RNG states")
        for key, value in asdict(config).items():
            if key not in ("epochs", "output", "resume", "device", "dataset_dir"):
                if saved["training_config"].get(key) != value:
                    raise ValueError(f"Resume mismatch: {key}")
        policy.load_state_dict(saved["policy_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        validation = TensorDict(saved["validation_data"], [config.val_size])
        history, first_epoch = saved["history"], saved["epoch"] + 1
        best_key, best_epoch = tuple(saved["best_key"]), saved["best_epoch"]
        best_weights, initial_metrics = saved["best_policy_state_dict"], saved["initial_metrics"]
        generator.rng.set_state(saved["generator_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if config.device.startswith("cuda") and saved["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
    if first_epoch > config.epochs:
        raise ValueError("epochs must exceed the saved completed epoch")

    def evaluate():
        # 验证过程不消耗训练采样的随机数序列。
        devices = (
            [torch.device(config.device).index or 0] if config.device.startswith("cuda") else []
        )
        with torch.random.fork_rng(devices=devices):
            set_seed(config.seed + 30000, device=config.device)
            results = []
            for start in range(0, config.val_size, config.batch_instances):
                batch = validation[start : start + config.batch_instances].to(config.device)
                results.append(
                    best_of_k(policy, env, batch, config.val_samples, config.sample_chunk)
                )
        merged = {
            k: torch.cat([r[k] for r in results])
            for k in ("cost", "reward", "infeasible", "failed_candidates")
        }
        merged.update(samples=config.val_samples, seconds=sum(r["seconds"] for r in results))
        return inference_metrics(merged)

    if initial_metrics is None:
        initial_metrics = evaluate()
        best_key = (initial_metrics["infeasible_rate"], -initial_metrics["reward"])

    def base_payload(weights, epoch):
        return {
            "format_version": 2,
            "architecture": "coupled_path_v1",
            "policy_config": policy.config,
            "policy_state_dict": weights,
            "epoch": epoch,
            "generator_config": generator_config,
            "env_config": {"penalty": config.penalty},
        }

    save_checkpoint(base_payload(best_weights, best_epoch), output / "best.npz")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    log = output / "metrics.jsonl"
    log.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8"
    )
    for epoch in range(first_epoch, config.epochs + 1):
        started = time.perf_counter()
        policy.train()
        training = (
            training_data if training_data is not None else generator([config.train_instances])
        )
        order = torch.randperm(config.train_instances)
        loss_sum, reward_sum, failures, all_failed, count = 0.0, 0.0, 0, 0, 0
        mixture = config.best_weight if epoch > config.best_warmup_epochs else 0.0
        for start in range(0, config.train_instances, config.batch_instances):
            batch = training[order[start : start + config.batch_instances]].to(config.device)
            size = batch.batch_size[0]
            result = policy(batch, env, num_samples=config.group_size, return_actions=False)
            shape = (size, config.group_size)
            loss, _ = grouped_rl_loss(
                result["reward"].reshape(shape),
                result["log_likelihood"].reshape(shape),
                result["entropy"].reshape(shape) / config.n_c,
                mixture,
                config.reward_scale,
                config.entropy_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite path RL loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * size
            reward_sum += float(result["reward"].detach().sum())
            failures += int(result["infeasible"].sum())
            all_failed += int(result["infeasible"].reshape(shape).all(-1).sum())
            count += size
            if (
                start == 0
                or count == config.train_instances
                or count % (8 * config.batch_instances) == 0
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "instances": count,
                            "trajectories": count * config.group_size,
                            "grad_norm": float(grad_norm),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        train_seconds = time.perf_counter() - started
        metrics = evaluate()
        trajectories = count * config.group_size
        row = {
            "epoch": epoch,
            "loss": loss_sum / count,
            "train_reward": reward_sum / trajectories,
            "train_infeasible_rate": failures / trajectories,
            "all_failed_group_rate": all_failed / count,
            "training_instances": count,
            "training_trajectories": trajectories,
            "best_weight": mixture,
            "train_seconds": train_seconds,
            "validation": metrics,
        }
        history.append(row)
        key = (metrics["infeasible_rate"], -metrics["reward"])
        if key < best_key:
            best_key, best_epoch, best_weights = key, epoch, _cpu_weights(policy)
            save_checkpoint(base_payload(best_weights, best_epoch), output / "best.npz")
        payload = base_payload(_cpu_weights(policy), epoch)
        payload.update(
            training_config=asdict(config),
            dataset_metadata=dataset_metadata,
            optimizer_state_dict=optimizer.state_dict(),
            validation_data={k: v.cpu() for k, v in validation.items()},
            generator_rng_state=generator.rng.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            best_key=best_key,
            best_epoch=best_epoch,
            best_policy_state_dict=best_weights,
            initial_metrics=initial_metrics,
            history=copy.deepcopy(history),
        )
        save_checkpoint(payload, output / "last.npz")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    report = {
        "architecture": "coupled_path_v1",
        "initial_metrics": initial_metrics,
        "best_epoch": best_epoch,
        "best_key": best_key,
        "history": history,
        "trainable_parameters": sum(p.numel() for p in policy.parameters()),
        "total_training_trajectories": len(history) * config.train_instances * config.group_size,
        "best_checkpoint": str((output / "best.npz").resolve()),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="保留分组路径强化学习和原生 RL4CO rollout 基线训练流程。"
    )
    parser.add_argument("--method", choices=("path", "two_stage"))
    defaults = {**asdict(TrainConfig()), **asdict(PathTrainConfig())}
    for name, value in defaults.items():
        options = {"default": None}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value) if value is not None else str
        parser.add_argument("--" + name.replace("_", "-"), **options)
    settings = parse_config(parser, argv)
    method = settings.pop("method", "path")
    cls = PathTrainConfig if method == "path" else TrainConfig
    if settings.get("resume"):
        payload = load_checkpoint(settings["resume"])
        expected = "path" if payload.get("architecture") == "coupled_path_v1" else "two_stage"
        if method != expected:
            raise ValueError("Resume method does not match checkpoint architecture")
        settings = {**payload["training_config"], **settings}
    if not settings.get("device") or "data_mode" not in settings:
        parser.error("Set device and data_mode explicitly using flags or a YAML config")
    unknown = set(settings) - set(asdict(cls()))
    if unknown:
        parser.error(f"Options do not apply to {method}: {sorted(unknown)}")
    (train_path if method == "path" else run_training)(cls(**settings))


@dataclass
class TrainConfig:
    epochs: int = 100
    train_size: int = 10000
    val_size: int = 256
    batch_size: int = 128
    n_w: int = 5
    n_s: int = 10
    n_c: int = 20
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    lr: float = 1e-4
    seed: int = 42
    device: str = ""
    data_mode: str = "synthetic"
    threads: int = 1
    penalty: float = 1e5
    output: str = "checkpoints/two_stage"
    resume: str | None = None
    dataset_dir: str | None = None
    initialize_from: str | None = None


def runtime_versions() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "tensordict",
            "torchrl",
            "rl4co",
            "lightning",
            "numpy",
            "scipy",
        )
    }
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def _tensor_data(td: TensorDict) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in td.items()}


def validation_metrics(
    policy: TSCFLPPolicy, env: TSCFLPEnv, data: TensorDict, batch_size: int, device: str
) -> dict[str, float | None]:
    policy.eval()
    states = []
    with torch.inference_mode():
        for start in range(0, data.batch_size[0], batch_size):
            batch = data[start : start + batch_size].to(device)
            states.append(policy(env.reset(batch), env, decode_type="greedy")["state"].cpu())
    return episode_metrics(torch.cat(states, 0))


def run_training(config: TrainConfig) -> dict[str, Any]:
    # 使用完整回合奖励训练两个解码器、聚合模块和编码器。
    # 检查点保存策略、基线及其验证数据、优化器、验证实例、生成器随机状态、
    # CPU/CUDA 随机状态和历史记录，支持在回合轮次边界精确续训。
    # 训练流程不导入 Gurobi；gap_to_gurobi 仅由评估流程填写。
    validate_training_paths(config)
    for name in ("epochs", "train_size", "val_size", "batch_size", "threads"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    validate_device(config.device)
    set_seed(config.seed, device=config.device)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    setup_started = time.perf_counter()
    validate_data_mode(config)
    training_data, dataset_metadata = None, None
    if config.data_mode == "npz":
        splits, dataset_metadata = load_dataset_directory(
            config.dataset_dir, splits=("train", "val")
        )
        training_data, validation = splits["train"], splits["val"]
        if (
            config.train_size != training_data.batch_size[0]
            or config.val_size != validation.batch_size[0]
        ):
            raise ValueError("File training requires the complete train and validation split sizes")
        actual = tuple(dataset_metadata["splits"]["train"][key] for key in ("n_w", "n_s", "n_c"))
        if actual != (config.n_w, config.n_s, config.n_c):
            raise ValueError("Dataset dimensions do not match the training configuration")
    generator_config = {
        "n_w": config.n_w,
        "n_s": config.n_s,
        "n_c": config.n_c,
        "cost_factor": 10.0,
    }
    gen = (
        FixedDatasetGenerator(validation, seed=config.seed, device=config.device)
        if training_data is not None
        else TSCFLPGenerator(**generator_config, seed=config.seed, device=config.device)
    )
    env = TSCFLPEnv(generator=gen, seed=config.seed, penalty=config.penalty)
    policy = TSCFLPPolicy(config.embed_dim, config.num_heads, config.num_layers).to(config.device)
    initialization = None
    if config.initialize_from and not config.resume:
        initial = load_checkpoint(config.initialize_from)
        weights = initial.get("best_policy_state_dict") or initial["policy_state_dict"]
        policy.load_state_dict(weights, strict=True)
        if initial.get("generator_config") == generator_config and "generator_rng_state" in initial:
            gen.rng.set_state(initial["generator_rng_state"])
        if "torch_rng_state" in initial:
            torch.set_rng_state(initial["torch_rng_state"])
        initialization = {
            "checkpoint": str(Path(config.initialize_from).resolve()),
            "selected_policy_epoch": initial.get("best_epoch", initial.get("epoch")),
            "source_completed_epochs": initial.get("epoch"),
            "source_training_episodes": len(initial.get("history", []))
            * initial.get("training_config", {}).get("train_size", 0),
            "optimizer": "fresh Adam; baseline rebuilt from initialized actor",
            "generator_stream": "continued when matching synthetic generator state is available",
            "note": "warm start permits larger data budgets; this is not exact checkpoint resume",
        }
    learner = REINFORCE(
        env,
        policy,
        baseline=RolloutBaseline(),
        reward_scale=1000,
        batch_size=config.batch_size,
        train_data_size=config.train_size,
        val_data_size=config.val_size,
    )
    learner.baseline.setup(
        policy,
        env,
        batch_size=config.batch_size,
        device=config.device,
        dataset_size=config.val_size,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    if training_data is None:
        validation = TSCFLPGenerator(**generator_config, seed=config.seed + 1)([config.val_size])
    initial_validation = validation_metrics(
        policy, env, validation, config.batch_size, config.device
    )
    initial_validation["objective_loss"] = -initial_validation["reward"]
    initial_validation["source"] = "same fixed validation instances as every training epoch"
    epoch_timings = []
    history, first_epoch, best_reward = [], 1, initial_validation["reward"]
    best_state, best_epoch = _tensor_data(policy.state_dict()), 0
    if config.resume:
        payload = load_checkpoint(config.resume)
        if payload.get("format_version") != 1 or "training_config" not in payload:
            raise ValueError(
                "Resume requires a training checkpoint, not an overfit/inference checkpoint"
            )
        previous = payload["training_config"]
        for name, value in asdict(config).items():
            if name not in (
                "epochs",
                "output",
                "resume",
                "device",
                "dataset_dir",
            ) and value != previous.get(name):
                raise ValueError(f"Resume configuration mismatch: {name}")
        if dataset_identity(payload["dataset_metadata"]) != dataset_identity(dataset_metadata):
            raise ValueError(
                "Resume source dataset changed (path, content hash, dimensions or metadata)"
            )
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        learner.baseline.load_state_dict(payload["baseline_state_dict"], strict=True)
        learner.baseline.policy.eval()
        learner.baseline.dataset = TensorDictDataset(
            TensorDict(payload["baseline_data"], [config.val_size])
        )
        learner.baseline.bl_vals = payload["baseline_values"].numpy()
        learner.baseline.mean = float(payload["baseline_mean"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        validation = TensorDict(payload["validation_data"], [config.val_size])
        history = payload["history"]
        initial_validation = payload.get("initial_validation", initial_validation)
        initialization = payload.get("initialization")
        epoch_timings = payload.get("epoch_timings", [])
        first_epoch, best_reward = payload["epoch"] + 1, payload["best_reward"]
        best_state = payload["best_policy_state_dict"]
        best_epoch = payload["best_epoch"]
        gen.rng.set_state(payload["generator_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if config.device.startswith("cuda") and payload["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    if first_epoch > config.epochs:
        raise ValueError("epochs must exceed the last completed checkpoint epoch")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)

    def save_best() -> None:
        save_checkpoint(
            {
                "format_version": 1,
                "epoch": best_epoch,
                "policy_config": policy.config,
                "generator_config": generator_config,
                "env_config": {"penalty": config.penalty},
                "policy_state_dict": best_state,
                "validation_reward": best_reward,
                "dataset_metadata": dataset_metadata,
                "initialization": initialization,
            },
            output / "best.npz",
        )

    if best_state is not None:
        save_best()  # 续训到其他目录时，仍保留已有最优策略快照。
    versions = runtime_versions()
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    if dataset_metadata:
        (output / "dataset_manifest.json").write_text(
            json.dumps(dataset_metadata, indent=2), encoding="utf-8"
        )
    setup_seconds = time.perf_counter() - setup_started
    # 续训时先写入恢复的历史记录，使新目录中的日志也保持完整。
    log_path = output / "metrics.jsonl"
    log_path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8"
    )

    for epoch in range(first_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        policy.train()
        training = training_data if training_data is not None else gen([config.train_size])
        ordering = torch.randperm(config.train_size)
        loss_sum, reward_sum, count = 0.0, 0.0, 0
        for start in range(0, config.train_size, config.batch_size):
            batch = training[ordering[start : start + config.batch_size].to(config.device)]
            td = env.reset(batch)
            result = policy(td, env, phase="train")
            result = learner.calculate_loss(td, batch, result)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            size = batch.batch_size[0]
            loss_sum += float(loss.detach()) * size
            reward_sum += float(result["reward"].sum())
            count += size
            if (
                start == 0
                or (start // config.batch_size + 1) % 10 == 0
                or count == config.train_size
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "trained": count,
                            "train_size": config.train_size,
                            "elapsed_seconds": time.perf_counter() - epoch_started,
                        }
                    ),
                    flush=True,
                )
        policy.eval()
        states = []
        with torch.inference_mode():
            for start in range(0, config.val_size, config.batch_size):
                batch = validation[start : start + config.batch_size].to(config.device)
                states.append(policy(env.reset(batch), env, phase="val")["state"].cpu())
        validation_state = torch.cat(states, 0)
        metrics = {
            "epoch": epoch,
            "reinforce_loss": loss_sum / count,
            "train_objective_loss": -reward_sum / count,
            **episode_metrics(validation_state),
            "gap_to_gurobi": None,
        }
        history.append(metrics)
        learner.baseline.epoch_callback(
            policy,
            env,
            batch_size=config.batch_size,
            device=config.device,
            epoch=epoch,
            dataset_size=config.val_size,
        )
        epoch_timings.append({"epoch": epoch, "seconds": time.perf_counter() - epoch_started})
        improved = metrics["reward"] > best_reward
        if improved:
            best_reward = metrics["reward"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
            }
        baseline_data = learner.baseline.dataset.collate_fn(learner.baseline.dataset.data)
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "policy_config": policy.config,
            "generator_config": generator_config,
            "training_config": asdict(config),
            "env_config": {"penalty": config.penalty},
            "policy_state_dict": policy.state_dict(),
            "baseline_state_dict": learner.baseline.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_data": _tensor_data(validation),
            "baseline_data": _tensor_data(baseline_data),
            "baseline_values": torch.from_numpy(learner.baseline.bl_vals.copy()),
            "baseline_mean": float(learner.baseline.mean),
            "generator_rng_state": gen.rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "best_reward": best_reward,
            "best_epoch": best_epoch,
            "best_policy_state_dict": best_state,
            "history": copy.deepcopy(history),
            "runtime": versions,
            "dataset_metadata": dataset_metadata,
            "initial_validation": initial_validation,
            "epoch_timings": epoch_timings,
            "initialization": initialization,
        }
        save_checkpoint(payload, output / "last.npz")
        if improved:
            save_best()
        print(json.dumps(metrics, allow_nan=False), flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
    report = {
        "history": history,
        "last_checkpoint": str(output / "last.npz"),
        "best_reward": best_reward,
        "best_epoch": best_epoch,
        "initial_validation": initial_validation,
        "dataset_metadata": dataset_metadata,
        "setup_seconds": setup_seconds,
        "epoch_timings": epoch_timings,
        "training_seconds": sum(row["seconds"] for row in epoch_timings),
        "initialization": initialization,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    main()

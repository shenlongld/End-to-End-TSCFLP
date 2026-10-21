# 分组路径强化学习：保持第145轮实验的损失、优化器与采样流程。

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

from .common import parse_config, set_seed, validate_device
from .dataset import dataset_identity, load_checkpoint, load_dataset_directory, save_checkpoint
from .env import Env
from .generator import TSCFLPGenerator
from .infer import best_of_k
from .metrics import inference_metrics
from .model import Policy


def check_mode(config):
    if config.data_mode not in ("synthetic", "npz"):
        raise ValueError("data_mode must be synthetic or npz")
    if config.data_mode == "npz":
        if config.dataset_dir is None:
            raise ValueError("npz training requires dataset_dir")
        Path(config.dataset_dir).resolve(strict=True)
    elif config.dataset_dir is not None:
        raise ValueError("dataset_dir is only valid for data_mode=npz")


def check_data(config, training, validation, train_size):
    if training.batch_size[0] != train_size or validation.batch_size[0] != config.val_size:
        raise ValueError("Training and validation sizes must match the complete NPZ splits")
    dimensions = tuple(
        training[key].shape[-1]
        for key in ("warehouse_capacity", "station_capacity", "customer_demand")
    )
    if dimensions != (config.n_w, config.n_s, config.n_c):
        raise ValueError("Dataset dimensions do not match the training configuration")


def check_paths(config):
    validate_device(config.device)
    check_mode(config)
    if config.resume is not None:
        path = Path(config.resume).resolve(strict=True)
        if path.suffix != ".npz":
            raise ValueError("Checkpoints must be pickle-free NPZ archives")
    if not config.resume and (Path(config.output) / "last.npz").exists():
        raise ValueError("Output already has a run; use resume or a new directory")


def group_loss(
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
class TrainConfig:
    epochs: int = 150
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


def train(config: TrainConfig):
    check_paths(config)
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
    check_mode(config)
    generator = TSCFLPGenerator(**generator_config, seed=config.seed + 10000)
    training_data, dataset_metadata = None, None
    if config.data_mode == "npz":
        splits, dataset_metadata = load_dataset_directory(config.dataset_dir, ("train", "val"))
        training_data, validation = splits["train"], splits["val"]
        check_data(config, training_data, validation, config.train_instances)
    else:
        validation = TSCFLPGenerator(**generator_config, seed=config.seed + 20000)(
            [config.val_size]
        )
    env = Env(penalty=config.penalty)
    policy = Policy(
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
            loss, _ = group_loss(
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

    parser = argparse.ArgumentParser(description="分组路径强化学习训练与精确断点续训。")
    for name, value in asdict(TrainConfig()).items():
        options = {"default": None}
        if isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value) if value is not None else str
        parser.add_argument("--" + name.replace("_", "-"), **options)
    settings = parse_config(parser, argv)
    if settings.get("resume"):
        saved = load_checkpoint(settings["resume"])
        if saved.get("format_version") != 2 or saved.get("architecture") != "coupled_path_v1":
            parser.error("Resume requires coupled_path_v1 checkpoint format 2")
        settings = {**saved["training_config"], **settings}
    if not settings.get("device") or "data_mode" not in settings:
        parser.error("Set device and data_mode using flags or a YAML config")
    train(TrainConfig(**settings))


if __name__ == "__main__":
    main()

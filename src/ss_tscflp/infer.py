# 贪心推理，以及分块执行的 K 候选择优采样推理。

import time

import torch

from .common import synchronize
from .env import Env
from .model import load_policy
from .problem import check_feasible, objective


@torch.inference_mode()
def best_of_k(policy, env, data, samples=64, sample_chunk=16, decode_type="sampling"):
    # 每个实例恰好生成 K 个候选，跨分块复用一次静态编码结果。
    # 优先从可行候选中择优；若某行的候选全部失败，则明确标记失败，
    # 并保留部分状态用于诊断，不将其作为可行解报告。
    if samples < 1 or sample_chunk < 1:
        raise ValueError("samples and sample_chunk must be positive")
    if decode_type not in ("sampling", "greedy"):
        raise ValueError("decode_type must be sampling or greedy")
    was_training = policy.training
    policy.eval()
    device = data["customer_demand"].device
    synchronize(device)
    started = time.perf_counter()
    embedding = policy.encoder(data)
    b = data.batch_size[0]
    best, best_rank = None, None
    failed_count = torch.zeros(b, dtype=torch.long, device=data["customer_demand"].device)
    try:
        for start in range(0, samples, sample_chunk):
            k = min(sample_chunk, samples - start)
            out = policy(
                data,
                env,
                decode_type=decode_type,
                num_samples=k,
                return_actions=False,
                embeddings=embedding,
            )
            failed = out["infeasible"].reshape(b, k)
            ranks = out["cost"].reshape(b, k).masked_fill(failed, torch.inf)
            selected = ranks.argmin(-1)
            indices = torch.arange(b, device=ranks.device) * k + selected
            candidate = out["state"][indices].clone()
            rank = ranks.gather(1, selected[:, None]).squeeze(1)
            failed_count += failed.sum(-1)
            if best is None:
                best, best_rank = candidate, rank
            else:
                replace = rank < best_rank
                best[replace] = candidate[replace]
                best_rank = torch.minimum(best_rank, rank)
    finally:
        policy.train(was_training)
    claimed_valid = ~best["infeasible"]
    if claimed_valid.any():
        check_feasible(best[claimed_valid], raise_on_error=True)
    valid = check_feasible(best).valid
    synchronize(device)
    return {
        "state": best,
        "cost": objective(best),
        "infeasible": ~valid,
        "reward": best["reward"],
        "failed_candidates": failed_count,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


def load_solver(checkpoint, device):
    from .common import validate_device

    validate_device(device)
    policy, payload = load_policy(checkpoint, device)
    env = Env(**payload["env_config"])
    return policy, env, payload


def infer_dataset(
    policy, env, data, *, device, batch_size=8, samples=64, sample_chunk=8, decode_type="sampling"
):
    from .common import validate_device

    validate_device(device)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    states, results = [], []
    for start in range(0, data.batch_size[0], batch_size):
        batch = data[start : start + batch_size].to(device)
        result = best_of_k(policy, env, batch, samples, sample_chunk, decode_type)
        # 首个决策前就全部失败的批次没有临时 action 字段。
        states.append(result["state"].exclude("action").cpu())
        results.append(result)
    merged = {
        key: torch.cat([r[key].cpu() for r in results])
        for key in ("cost", "reward", "infeasible", "failed_candidates")
    }
    merged.update(
        state=torch.cat(states), samples=samples, seconds=sum(r["seconds"] for r in results)
    )
    merged["batch_seconds"] = [r["seconds"] for r in results]
    return merged


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    from .baselines.common import save_solutions
    from .common import parse_config, set_seed
    from .dataset import file_sha256, load_instances
    from .metrics import inference_metrics, timing_metrics

    parser = argparse.ArgumentParser(description="贪心推理，以及分块执行的 K 候选择优采样推理。")
    for name in ("checkpoint", "data", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--decode-type", choices=("greedy", "sampling"))
    parser.add_argument("--temperature", type=float)
    for name in ("samples", "sample_chunk", "batch_size", "seed", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parse_config(parser, argv)
    for key in ("checkpoint", "data", "output", "device"):
        if key not in args:
            parser.error(f"Missing required setting: {key}")
    device = args["device"]
    set_seed(args.get("seed", 42), device=device)
    torch.set_num_threads(args.get("threads", 1))
    data, metadata = load_instances(args["data"])
    policy, env, _ = load_solver(args["checkpoint"], device)
    if args.get("temperature") is not None:
        import math

        if not math.isfinite(args["temperature"]) or args["temperature"] <= 0:
            raise ValueError("temperature must be finite and positive")
        policy.temperature = args["temperature"]
    mode = args.get("decode_type", "sampling")
    samples = args.get("samples", 1 if mode == "greedy" else 64)
    result = infer_dataset(
        policy,
        env,
        data,
        device=device,
        batch_size=args.get("batch_size", 8),
        samples=samples,
        sample_chunk=args.get("sample_chunk", 8),
        decode_type=mode,
    )
    root = Path(args["output"])
    root.mkdir(parents=True, exist_ok=True)
    save_solutions(data, result["state"], root / "solutions.npz", metadata=metadata)
    report = {
        "method": mode,
        "dataset": metadata,
        "device": device,
        "seed": args.get("seed", 42),
        "temperature": policy.temperature,
        "checkpoint_sha256": file_sha256(args["checkpoint"]),
        **inference_metrics(result),
        **timing_metrics(result["seconds"], data.batch_size[0], result["batch_seconds"]),
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

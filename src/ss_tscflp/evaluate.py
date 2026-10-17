# 在同一份显式指定的 NPZ 测试集上比较神经网络、模拟退火和 Gurobi。

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .baselines.common import save_solutions
from .common import parse_config, set_seed
from .dataset import file_sha256, load_instances
from .infer import infer_dataset, load_solver
from .metrics import summarize_costs, timing_metrics
from .problem import Solution, check_feasible, compute_gap, objective


def evaluate(
    *,
    data,
    output,
    device,
    checkpoint=None,
    methods=("greedy", "sampling", "sa"),
    seed=42,
    threads=1,
    batch_size=8,
    samples=64,
    sample_chunk=8,
    sa_iterations=5000,
    sa_restarts=4,
    time_limit=60.0,
    mip_gap=0.0,
):
    methods = list(methods)
    allowed = {"greedy", "sampling", "sa", "gurobi"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - allowed:
        raise ValueError("Choose distinct methods from greedy, sampling, sa, gurobi")
    if min(threads, batch_size, samples, sample_chunk, sa_iterations, sa_restarts) < 1:
        raise ValueError("All thread, batch, candidate and SA budgets must be positive")
    set_seed(seed, device=device)
    torch.set_num_threads(threads)
    instances, metadata = load_instances(data)
    count = instances.batch_size[0]
    if {"greedy", "sampling"} & set(methods):
        if checkpoint is None:
            raise ValueError("Neural methods require an explicit checkpoint")
        policy, env, payload = load_solver(checkpoint, device)
    elif checkpoint is not None:
        raise ValueError("checkpoint is only used by neural methods")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    results, summaries, arrays = [], {}, {}
    references = torch.full((count,), torch.nan, dtype=torch.float64)
    for method in methods:
        set_seed(seed, device=device)
        if method in ("greedy", "sampling"):
            result = infer_dataset(
                policy,
                env,
                instances,
                device=device,
                batch_size=batch_size,
                samples=1 if method == "greedy" else samples,
                sample_chunk=sample_chunk,
                decode_type=method,
            )
            state = result["state"]
            costs, valid = objective(state).double(), check_feasible(state).valid
            summary = timing_metrics(result["seconds"], count, result["batch_seconds"])
            summary.update(
                seconds=result["seconds"],
                samples=result["samples"],
                candidate_infeasible_rate=float(result["failed_candidates"].sum())
                / (count * result["samples"]),
            )
            records = [{} for _ in range(count)]
        else:
            if method == "sa":
                from .baselines.sa import solve_simulated_annealing

                solve = solve_simulated_annealing
                options = dict(iterations=sa_iterations, restarts=sa_restarts)
            else:
                from .baselines.gurobi import solve_single_source_milp

                solve = solve_single_source_milp
                options = dict(time_limit=time_limit, mip_gap=mip_gap, threads=threads)
            records, states = [], []
            started = time.perf_counter()
            for index in range(count):
                instance = instances[index : index + 1]
                result = solve(instance, seed=seed + index, **options)
                records.append(
                    {
                        key: value
                        for key, value in vars(result).items()
                        if key not in ("solution", "z", "a")
                    }
                )
                if result.solution is None:
                    solution = Solution(
                        torch.full_like(instance["customer_demand"], -1, dtype=torch.long),
                        torch.full_like(instance["station_capacity"], -1, dtype=torch.long),
                    )
                else:
                    solution = Solution(
                        result.solution["customer_station_assignment"],
                        result.solution["station_warehouse_assignment"],
                    )
                    check_feasible(result.solution, raise_on_error=True)
                states.append(solution.attach(instance))
                if method == "gurobi" and result.is_optimal:
                    references[index] = result.optimal_cost
            elapsed = time.perf_counter() - started
            state = torch.cat(states)
            costs, valid = objective(state).double(), check_feasible(state).valid
            summary = {"seconds": elapsed, **timing_metrics(elapsed, count)}
        save_solutions(instances, state, root / f"{method}.npz", metadata=metadata)
        arrays[method] = (costs, valid)
        summaries[method] = summary
        for index in range(count):
            results.append(
                {
                    "instance": index,
                    "method": method,
                    **records[index],
                    "feasible": bool(valid[index]),
                    "cost": float(costs[index]) if valid[index] else None,
                }
            )
    for method, (costs, valid) in arrays.items():
        summaries[method].update(summarize_costs(costs, valid, references))
    for record in results:
        index, method = record["instance"], record["method"]
        costs, valid = arrays[method]
        gap = float(compute_gap(costs, references)[index])
        record["gap_percent"] = (
            (
                gap
                if torch.isfinite(torch.tensor(gap))
                else "infinity"
                if gap == float("inf")
                else None
            )
            if valid[index]
            else None
        )
    report = {
        "dataset": metadata,
        "device": device,
        "seed": seed,
        "threads": threads,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "proven_optima": int(torch.isfinite(references).sum()),
        "budgets": {
            "samples": samples,
            "sample_chunk": sample_chunk,
            "batch_size": batch_size,
            "sa_iterations": sa_iterations,
            "sa_restarts": sa_restarts,
            "time_limit": time_limit,
            "mip_gap": mip_gap,
        },
        "methods": summaries,
        "results": results,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("instance", "method", "feasible", "cost", "gap_percent"),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="在同一份显式指定的 NPZ 测试集上比较神经网络、模拟退火和 Gurobi。"
    )
    for name in ("data", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--methods", nargs="+", choices=("greedy", "sampling", "sa", "gurobi"))
    for name in (
        "seed",
        "threads",
        "batch_size",
        "samples",
        "sample_chunk",
        "sa_iterations",
        "sa_restarts",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-gap", type=float)
    args = parse_config(parser, argv)
    if any(key not in args for key in ("data", "output", "device")):
        parser.error("data, output and device are required")
    report = evaluate(**args)
    print(
        json.dumps(
            {"proven_optima": report["proven_optima"], "methods": report["methods"]},
            allow_nan=False,
        )
    )


def benchmark_main(argv=None):
    parser = argparse.ArgumentParser(description="同步测量推理吞吐量和批次延迟")
    for name in ("data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--decode-type", choices=("greedy", "sampling"), default="greedy")
    for name, default in (
        ("warmup", 1),
        ("repeats", 5),
        ("batch_size", 1),
        ("samples", 64),
        ("sample_chunk", 8),
        ("seed", 42),
        ("threads", 1),
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args(argv)
    if (
        args.warmup < 0
        or min(args.repeats, args.batch_size, args.samples, args.sample_chunk, args.threads) < 1
    ):
        parser.error("warmup must be nonnegative and other budgets positive")
    set_seed(args.seed, device=args.device)
    torch.set_num_threads(args.threads)
    data, metadata = load_instances(args.data)
    policy, env, _ = load_solver(args.checkpoint, args.device)
    elapsed, latencies = [], []
    for index in range(args.warmup + args.repeats):
        result = infer_dataset(
            policy,
            env,
            data,
            device=args.device,
            batch_size=args.batch_size,
            samples=1 if args.decode_type == "greedy" else args.samples,
            sample_chunk=args.sample_chunk,
            decode_type=args.decode_type,
        )
        if index >= args.warmup:
            elapsed.append(result["seconds"])
            latencies.extend(result["batch_seconds"])
    from .train import runtime_versions

    report = {
        "dataset": metadata,
        "runtime": runtime_versions(),
        "device": args.device,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "decode_type": args.decode_type,
        "samples": result["samples"],
        "sample_chunk": args.sample_chunk,
        "timing_scope": (
            "encoder, decoding, canonical validation and device synchronization; "
            "excludes I/O and host-device transfer"
        ),
        "run_seconds": elapsed,
        **timing_metrics(sum(elapsed), len(data) * args.repeats, latencies),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

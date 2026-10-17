# 分配方案的交换格式，以及基线共用的命令行处理。

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..common import parse_config, set_seed
from ..dataset import load_instances
from ..metrics import summarize_costs, timing_metrics
from ..problem import Solution, check_feasible, objective


def save_solutions(data, solution, path, *, metadata=None):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Solution path must end with .npz")
    report = check_feasible(solution)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            customer_station_assignment=solution["customer_station_assignment"].cpu().numpy(),
            station_warehouse_assignment=solution["station_warehouse_assignment"].cpu().numpy(),
            feasible=report.valid.cpu().numpy(),
            cost=objective(
                data,
                Solution(
                    solution["customer_station_assignment"],
                    solution["station_warehouse_assignment"],
                ),
            )
            .cpu()
            .numpy(),
            metadata_json=np.array(json.dumps(metadata or {}, allow_nan=False)),
        )


def load_solutions(data, path):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Solution path must end with .npz")
    with np.load(path, allow_pickle=False) as archive:
        solution = Solution(
            torch.from_numpy(archive["customer_station_assignment"].copy()),
            torch.from_numpy(archive["station_warehouse_assignment"].copy()),
        ).attach(data)
        metadata = json.loads(str(archive["metadata_json"]))
    return solution, metadata


def baseline_main(kind, solve, argv=None):
    parser = argparse.ArgumentParser(description=f"{kind} 基线求解与统一分配方案检查")
    for name in ("data", "output", "solution"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int)
    if kind == "sa":
        parser.add_argument("--iterations", type=int)
        parser.add_argument("--restarts", type=int)
    else:
        parser.add_argument("--time-limit", type=float)
        parser.add_argument("--mip-gap", type=float)
    args = parse_config(parser, argv)
    if "data" not in args or "device" not in args:
        parser.error("data and device are required")
    if args["device"] != "cpu":
        raise ValueError("The SA/Gurobi interface requires explicit device=cpu")
    set_seed(args.get("seed", 42), device="cpu")
    torch.set_num_threads(args.get("threads", 1))
    data, metadata = load_instances(args["data"])
    if args.get("solution"):
        state, saved = load_solutions(data, args["solution"])
        if saved["sha256"] != metadata["sha256"]:
            raise ValueError("Solution dataset digest does not match the requested data")
        check_feasible(state, raise_on_error=True)
        report = {"valid": True, "cost": objective(state).tolist(), "dataset": metadata}
        print(json.dumps(report, allow_nan=False))
        return report
    if "output" not in args:
        parser.error("output is required when solving")
    root = Path(args["output"])
    options = (
        {"iterations": args.get("iterations", 5000), "restarts": args.get("restarts", 4)}
        if kind == "sa"
        else {
            "time_limit": args.get("time_limit", 60.0),
            "mip_gap": args.get("mip_gap", 0.0),
            "threads": args.get("threads", 1),
        }
    )
    import time

    started = time.perf_counter()
    records, states, costs, valid = [], [], [], []
    for index in range(data.batch_size[0]):
        result = solve(data[index : index + 1], seed=args.get("seed", 42) + index, **options)
        row = {
            key: value for key, value in vars(result).items() if key not in ("solution", "z", "a")
        }
        records.append(row)
        if result.solution is None:
            state = Solution(
                torch.full_like(data["customer_demand"][index : index + 1], -1, dtype=torch.long),
                torch.full_like(data["station_capacity"][index : index + 1], -1, dtype=torch.long),
            ).attach(data[index : index + 1])
        else:
            state = result.solution
        states.append(
            Solution(
                state["customer_station_assignment"], state["station_warehouse_assignment"]
            ).attach(data[index : index + 1])
        )
        valid.append(check_feasible(state).all_valid)
        costs.append(float(objective(state)[0]))
    seconds = time.perf_counter() - started
    save_solutions(data, torch.cat(states), root / "solutions.npz", metadata=metadata)
    report = {
        "method": kind,
        "dataset": metadata,
        "results": records,
        **summarize_costs(costs, valid),
        **timing_metrics(seconds, len(records)),
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, allow_nan=False))
    return report

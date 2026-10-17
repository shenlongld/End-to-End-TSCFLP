# 回合统计，以及经过显式设备同步的延迟和吞吐量指标。

import torch
from tensordict import TensorDict

from .problem import check_feasible, compute_gap, objective


def cost_consistency_error(td: TensorDict) -> torch.Tensor:
    return (td["incremental_cost"] - objective(td)).abs()


def episode_metrics(td: TensorDict) -> dict[str, float | None]:
    feasible = check_feasible(td).valid

    def mean_success(value: torch.Tensor) -> float | None:
        return float(value[feasible].float().mean()) if feasible.any() else None

    return {
        "cost": mean_success(objective(td)),
        "infeasible_rate": float(td["infeasible"].float().mean()),
        "active_stations": mean_success(td["station_open"].sum(-1)),
        "active_warehouses": mean_success(td["warehouse_open"].sum(-1)),
        "reward": float(td["reward"].mean()),
    }


def inference_metrics(result):
    feasible = ~result["infeasible"]
    return {
        "reward": float(result["reward"].mean()),
        "feasible_cost": float(result["cost"][feasible].mean()) if feasible.any() else None,
        "infeasible_rate": float((~feasible).float().mean()),
        "candidate_infeasible_rate": float(
            result["failed_candidates"].float().mean() / result["samples"]
        ),
        "samples": result["samples"],
        "seconds": result["seconds"],
    }


def timing_metrics(seconds, count, batch_seconds=()):
    if seconds <= 0 or count < 1:
        raise ValueError("Positive elapsed seconds and instance count required")
    batches = torch.tensor(list(batch_seconds), dtype=torch.float64)
    return {
        "instances_per_second": count / seconds,
        "amortized_ms_per_instance": 1000 * seconds / count,
        "batch_latency_p50_ms": float(torch.quantile(batches, 0.5) * 1000)
        if batches.numel()
        else None,
        "batch_latency_p95_ms": float(torch.quantile(batches, 0.95) * 1000)
        if batches.numel()
        else None,
    }


def summarize_costs(costs, valid, references=None):
    costs = torch.as_tensor(costs, dtype=torch.float64)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    gaps = compute_gap(
        costs, torch.full_like(costs, torch.nan) if references is None else references
    )
    available = valid & ~torch.isnan(gaps)
    mean_gap = float(gaps[available].mean()) if available.any() else None
    if mean_gap is not None and not torch.isfinite(torch.tensor(mean_gap)):
        mean_gap = "infinity"
    return {
        "feasible_cost": float(costs[valid].mean()) if valid.any() else None,
        "infeasible_rate": float((~valid).double().mean()),
        "gap_percent": mean_gap,
        "gap_instances": int(available.sum()),
    }

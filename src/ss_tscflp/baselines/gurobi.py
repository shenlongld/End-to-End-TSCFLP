# 单源问题的精确混合整数线性规划，用于测试、评估和独立求解。
# 当 a 为二元变量时，连续辅助变量 t[i,j] 严格等于 D[j] * a[i,j]，
# 并不表示可拆分的运输决策；返回的分配始终为二元矩阵或整数索引。

import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from ..problem import STATIC_KEYS, check_feasible, objective, validate_instance


class GurobiUnavailableError(RuntimeError):
    # Gurobi 未安装或许可证不可用时抛出此异常，不切换到其他求解器。
    pass


@dataclass(frozen=True)
class MILPResult:
    status: int
    is_optimal: bool
    optimal_cost: float | None
    objective_cost: float | None
    lower_bound: float | None
    mip_gap: float | None
    solve_time: float
    z: torch.Tensor | None
    a: torch.Tensor | None
    solution: TensorDict | None


def solve_single_source_milp(
    td: TensorDict,
    time_limit: float = 60.0,
    mip_gap: float = 0.0,
    seed: int = 0,
    threads: int = 1,
    verbose: bool = False,
) -> MILPResult:
    # 求解批量形状为 [1] 的单个实例，区分已证明最优解和限时内的当前可行解。
    validate_instance(td)
    if td.batch_size[0] != 1:
        raise ValueError("MILP accepts one instance with batch shape [1]")
    if not math.isfinite(time_limit) or time_limit <= 0 or not 0 <= mip_gap < 1 or threads < 1:
        raise ValueError("Invalid time_limit, mip_gap or threads")
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise GurobiUnavailableError(
            "Install evaluation dependency: pip install .[gurobi]"
        ) from exc
    try:
        gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam("OutputFlag", int(verbose))
        gurobi_env.start()
    except gp.GurobiError as exc:
        raise GurobiUnavailableError(f"Gurobi license unavailable: {exc}") from exc

    data = {
        key: value[0].detach().double().cpu().numpy()
        for key, value in td.items()
        if isinstance(value, torch.Tensor)
    }
    d, qs, qw = data["customer_demand"], data["station_capacity"], data["warehouse_capacity"]
    nw, ns, nc = len(qw), len(qs), len(d)
    with gurobi_env, gp.Model("single_source_tscflp", env=gurobi_env) as model:
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = mip_gap
        model.Params.Seed = seed
        model.Params.Threads = threads
        z = model.addVars(ns, nc, vtype=gp.GRB.BINARY, name="z")
        a = model.addVars(nw, ns, vtype=gp.GRB.BINARY, name="a")
        y = model.addVars(ns, vtype=gp.GRB.BINARY, name="y")
        x = model.addVars(nw, vtype=gp.GRB.BINARY, name="x")
        load = model.addVars(ns, lb=0, name="D")
        product = model.addVars(nw, ns, lb=0, name="D_times_a")
        for k in range(nc):
            model.addConstr(gp.quicksum(z[j, k] for j in range(ns)) == 1)
        for j in range(ns):
            model.addConstr(load[j] == gp.quicksum(float(d[k]) * z[j, k] for k in range(nc)))
            model.addConstr(load[j] <= float(qs[j]) * y[j])
            model.addConstr(y[j] <= gp.quicksum(z[j, k] for k in range(nc)))
            for k in range(nc):
                model.addConstr(z[j, k] <= y[j])
            model.addConstr(gp.quicksum(a[i, j] for i in range(nw)) == y[j])
            big_m = float(min(qs[j], d.sum()))
            for i in range(nw):
                model.addConstr(product[i, j] <= load[j])
                model.addConstr(product[i, j] <= big_m * a[i, j])
                model.addConstr(product[i, j] >= load[j] - big_m * (1 - a[i, j]))
        for i in range(nw):
            assigned = gp.quicksum(a[i, j] for j in range(ns))
            model.addConstr(gp.quicksum(product[i, j] for j in range(ns)) <= float(qw[i]) * x[i])
            model.addConstr(assigned <= ns * x[i])
            model.addConstr(x[i] <= assigned)
        model.setObjective(
            objective(
                td,
                terms={
                    "warehouse_fixed_cost": {(i,): x[i] for i in range(nw)},
                    "station_fixed_cost": {(j,): y[j] for j in range(ns)},
                    "cost_sc": {
                        (j, k): float(d[k]) * z[j, k] for j in range(ns) for k in range(nc)
                    },
                    "cost_ws": {(i, j): product[i, j] for i in range(nw) for j in range(ns)},
                },
            ),
            gp.GRB.MINIMIZE,
        )
        model.optimize()
        optimal = model.Status == gp.GRB.OPTIMAL and model.MIPGap <= 1e-9
        has_solution = model.SolCount > 0
        solver_cost = float(model.ObjVal) if has_solution else None
        bound = (
            float(model.ObjBound)
            if model.Status not in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD)
            else None
        )
        if bound is not None and not math.isfinite(bound):
            bound = None
        gap = float(model.MIPGap) if has_solution else None
        if gap is not None and not math.isfinite(gap):
            gap = None
        z_tensor = a_tensor = solution = None
        if has_solution:
            z_tensor = torch.tensor(
                [[round(z[j, k].X) for k in range(nc)] for j in range(ns)], dtype=torch.int64
            )
            a_tensor = torch.tensor(
                [[round(a[i, j].X) for j in range(ns)] for i in range(nw)], dtype=torch.int64
            )
            solution = td.select(*(key for key in STATIC_KEYS if key in td)).detach().cpu().clone()
            solution["customer_station_assignment"] = z_tensor.argmax(0).unsqueeze(0)
            sw = a_tensor.argmax(0).masked_fill(a_tensor.sum(0) == 0, -1)
            solution["station_warehouse_assignment"] = sw.unsqueeze(0)
            # 独立检查从求解器提取的分配，不能直接信任求解器输出。
            check_feasible(solution, raise_on_error=True)
            canonical_cost = float(objective(solution, dtype=torch.float64)[0])
            if not math.isclose(canonical_cost, solver_cost, rel_tol=1e-8, abs_tol=1e-5):
                raise RuntimeError("Gurobi objective differs from problem.objective")
            solver_cost = canonical_cost
        return MILPResult(
            status=int(model.Status),
            is_optimal=optimal,
            optimal_cost=solver_cost if optimal else None,
            objective_cost=solver_cost,
            lower_bound=bound,
            mip_gap=gap,
            solve_time=float(model.Runtime),
            z=z_tensor,
            a=a_tensor,
            solution=solution,
        )


def main(argv=None):
    from .common import baseline_main

    baseline_main("gurobi", solve_single_source_milp, argv)


if __name__ == "__main__":
    main()

# 保留原始实现中的科研回归测试。

from itertools import product

import pytest
import torch
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance
from test_problem import tiny_solution as tiny_solution

from ss_tscflp.baselines.gurobi import GurobiUnavailableError, solve_single_source_milp
from ss_tscflp.problem import (
    check_feasible,
    objective,
)


def test_assignment_checker_does_not_invoke_solver(tmp_path, tiny_instance, tiny_solution):
    from ss_tscflp.baselines.common import baseline_main, save_solutions
    from ss_tscflp.dataset import load_instances, save_instances

    data_path, solution_path = tmp_path / "data.npz", tmp_path / "solution.npz"
    save_instances(tiny_instance, data_path)
    data, metadata = load_instances(data_path)
    save_solutions(data, tiny_solution, solution_path, metadata=metadata)

    def forbidden(*args, **kwargs):
        raise AssertionError("Checker must not invoke an optimizer")

    report = baseline_main(
        "gurobi",
        forbidden,
        ["--data", str(data_path), "--solution", str(solution_path), "--device", "cpu"],
    )
    assert report["valid"] is True and report["cost"] == [120.0]


def milp_licensed_solve(td, time_limit=15):
    pytest.importorskip("gurobipy")
    try:
        return solve_single_source_milp(td, time_limit=time_limit)
    except GurobiUnavailableError as exc:
        pytest.skip(str(exc))


def milp_brute_force(td):
    best = float("inf")
    ns, nw, nc = (
        td[key].shape[-1] for key in ("station_capacity", "warehouse_capacity", "customer_demand")
    )
    for cs in product(range(ns), repeat=nc):
        active = sorted(set(cs))
        for wa in product(range(nw), repeat=len(active)):
            candidate = td.clone()
            candidate["customer_station_assignment"] = torch.tensor([cs])
            sw = [-1] * ns
            for station, warehouse in zip(active, wa, strict=True):
                sw[station] = warehouse
            candidate["station_warehouse_assignment"] = torch.tensor([sw])
            if check_feasible(candidate).all_valid:
                best = min(best, objective(candidate).item())
    return best


def test_milp_milp_equals_exhaustive_atomic_solution(tiny_instance):
    result = milp_licensed_solve(tiny_instance)
    assert result.is_optimal
    assert result.optimal_cost == pytest.approx(milp_brute_force(tiny_instance), abs=1e-4)
    assert check_feasible(result.solution).all_valid
    assert objective(result.solution).item() == pytest.approx(result.optimal_cost, abs=1e-4)
    assert (result.z.sum(0) == 1).all()
    assert result.solve_time >= 0


def test_milp_milp_capacity_and_zero_fixed_activation(tiny_instance):
    tiny_instance["station_fixed_cost"].zero_()
    tiny_instance["warehouse_fixed_cost"].zero_()
    tiny_instance["station_capacity"] = torch.tensor([[5.0, 4.0, 3.0]])
    tiny_instance["warehouse_capacity"] = torch.tensor([[5.0, 4.0]])
    result = milp_licensed_solve(tiny_instance)
    assert result.is_optimal
    assert result.optimal_cost == pytest.approx(milp_brute_force(tiny_instance), abs=1e-4)


def test_milp_milp_infeasible_has_no_fake_optimum(tiny_instance):
    tiny_instance["warehouse_capacity"].zero_()
    result = milp_licensed_solve(tiny_instance)
    assert not result.is_optimal
    assert result.optimal_cost is None and result.solution is None


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_milp_milp_timeout_does_not_claim_optimum(tiny_instance, monkeypatch, has_incumbent):
    # 微型问题可能在检查正数时间上限前就求解完成。
    # 测试固定终止状态元数据，避免依赖计时竞争；
    # 约束和返回的可行解仍由原生模型产生。
    gp = pytest.importorskip("gurobipy")
    native_model = gp.Model

    class TimedOutModel:
        def __init__(self, *args, **kwargs):
            self.model = native_model(*args, **kwargs)

        def __enter__(self):
            self.model.__enter__()
            return self

        def __exit__(self, *args):
            return self.model.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.model, name)

        @property
        def Status(self):
            return gp.GRB.TIME_LIMIT

        @property
        def SolCount(self):
            return self.model.SolCount if has_incumbent else 0

    monkeypatch.setattr(gp, "Model", TimedOutModel)
    result = milp_licensed_solve(tiny_instance)
    assert result.status == gp.GRB.TIME_LIMIT
    assert not result.is_optimal
    assert result.optimal_cost is None
    if has_incumbent:
        assert result.objective_cost == pytest.approx(milp_brute_force(tiny_instance), abs=1e-4)
        assert check_feasible(result.solution).all_valid
    else:
        assert result.objective_cost is None and result.solution is None


def test_milp_milp_accepts_a_completed_state_without_trusting_caches(tiny_solution):
    # 应根据求解器自身输出的分配检查结果，
    # 不能沿用其他解码方案的负载缓存。
    tiny_solution["station_load"].zero_()
    result = milp_licensed_solve(tiny_solution)
    assert result.is_optimal
    assert "station_load" not in result.solution
    assert check_feasible(result.solution).all_valid

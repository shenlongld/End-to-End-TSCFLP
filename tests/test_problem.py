# 保留原始实现中的科研回归测试。

import pytest
import torch
from tensordict import TensorDict

from ss_tscflp.common import set_seed
from ss_tscflp.env import TSCFLPEnv
from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.metrics import cost_consistency_error
from ss_tscflp.problem import (
    assignment_matrices,
    check_feasible,
    objective,
)


def test_public_dataclasses_use_canonical_functions(tiny_instance):
    from ss_tscflp.problem import Instance, Solution

    instance = Instance(tiny_instance)
    solution = Solution(torch.tensor([[0, 1, 0]]), torch.tensor([[0, 0, -1]]))
    assert instance.dimensions == (2, 3, 3)
    assert check_feasible(instance, solution).all_valid
    assert objective(instance, solution).item() == 120


def test_gap_missing_zero_and_exact_optimum():
    from ss_tscflp.problem import compute_gap

    cost = torch.tensor([12.0, 0.0, 1.0, 4.0, 5.0, 5.0])
    reference = torch.tensor([10.0, 0.0, 0.0, float("nan"), -1.0, 5.0])
    gap = compute_gap(cost, reference)
    assert gap[:3].tolist() == [20.0, 0.0, float("inf")]
    assert torch.isnan(gap[3:5]).all()
    assert gap[5].item() == 0


def test_canonical_definitions_and_pickle_free_production_sources():
    import ast
    from pathlib import Path

    import ss_tscflp

    root = Path(ss_tscflp.__file__).parent
    definitions = {name: [] for name in ("objective", "check_feasible", "compute_gap")}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in definitions:
                definitions[node.name].append(path.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                    assert node.func.attr not in ("load", "save")
    assert all(files == ["problem.py"] for files in definitions.values())


@pytest.fixture(autouse=True)
def deterministic_cpu():
    set_seed(1234, device="cpu")
    torch.set_num_threads(1)


@pytest.fixture
def tiny_instance():
    return TensorDict(
        {
            "customer_demand": torch.tensor([[4.0, 3.0, 2.0]]),
            "station_capacity": torch.tensor([[6.0, 5.0, 8.0]]),
            "warehouse_capacity": torch.tensor([[9.0, 9.0]]),
            "station_fixed_cost": torch.tensor([[20.0, 25.0, 30.0]]),
            "warehouse_fixed_cost": torch.tensor([[50.0, 70.0]]),
            "cost_sc": torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]]),
            "cost_ws": torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]]),
        },
        batch_size=[1],
    )


@pytest.fixture
def tiny_solution(tiny_instance):
    td = tiny_instance.clone()
    td["customer_station_assignment"] = torch.tensor([[0, 1, 0]])
    td["station_warehouse_assignment"] = torch.tensor([[0, 0, -1]])
    td["station_load"] = torch.tensor([[6.0, 3.0, 0.0]])
    td["warehouse_load"] = torch.tensor([[9.0, 0.0]])
    td["station_open"] = torch.tensor([[True, True, False]])
    td["warehouse_open"] = torch.tensor([[True, False]])
    return td


def test_cost_consistency_cost_consistency(tiny_instance):
    env = TSCFLPEnv()
    td = env.reset(tiny_instance)
    for action in [0, 1, 0, 0, 0]:
        td["action"] = torch.tensor([action])
        td = env.step(td)["next"]
        assert (cost_consistency_error(td) < 1e-4).all()
    assert objective(td).item() == 120


def test_cost_consistency_random_cost_consistency_across_scales():
    for nw, ns, nc in [(2, 4, 8), (5, 10, 20), (8, 15, 50)]:
        env = TSCFLPEnv(generator=TSCFLPGenerator(nw, ns, nc, seed=34))
        td = env.reset(batch_size=[32])
        while not td["done"].all():
            weights = td["action_mask"].float()
            weights[td["done"], 0] = 1
            td["action"] = torch.multinomial(weights, 1).squeeze(-1)
            td = env.step(td)["next"]
            assert (cost_consistency_error(td) < 1e-4).all()
        assert (td["reward"][~td["infeasible"]] == -objective(td)[~td["infeasible"]]).all()


def test_validity_checker_every_customer_exactly_one_station(tiny_solution):
    assert check_feasible(tiny_solution).all_valid
    z, _ = assignment_matrices(tiny_solution)
    assert (z.sum(1) == 1).all()


def test_validity_checker_every_active_station_exactly_one_warehouse(tiny_solution):
    _, a = assignment_matrices(tiny_solution)
    assert torch.equal(a.sum(1), tiny_solution["station_open"].long())


def test_validity_checker_station_capacity(tiny_solution):
    assert (
        check_feasible(tiny_solution).station_load <= tiny_solution["station_capacity"] + 1e-5
    ).all()
    tiny_solution["station_capacity"][0, 0] = 5
    assert check_feasible(tiny_solution).violations["station_capacity"].item()


def test_validity_checker_warehouse_capacity(tiny_solution):
    assert (
        check_feasible(tiny_solution).warehouse_load <= tiny_solution["warehouse_capacity"] + 1e-5
    ).all()
    tiny_solution["warehouse_capacity"][0, 0] = 8
    assert check_feasible(tiny_solution).violations["warehouse_capacity"].item()


@pytest.mark.parametrize(
    "key,index,value",
    [
        ("customer_station_assignment", 0, -1),
        ("customer_station_assignment", 0, 10),
        ("station_warehouse_assignment", 0, -1),
        ("station_warehouse_assignment", 0, 7),
        ("station_warehouse_assignment", 2, 1),
        ("station_load", 0, 1),
        ("station_open", 2, True),
        ("warehouse_open", 1, True),
    ],
)
def test_validity_checker_checker_rejects_invalid_solution(tiny_solution, key, index, value):
    tiny_solution[key][0, index] = value
    assert not check_feasible(tiny_solution).all_valid
    with pytest.raises(ValueError, match="Invalid TSCFLP"):
        check_feasible(tiny_solution, raise_on_error=True)


def test_validity_checker_checker_rejects_fractional_assignments(tiny_solution):
    tiny_solution["customer_station_assignment"] = tiny_solution[
        "customer_station_assignment"
    ].float()
    with pytest.raises(ValueError, match="int64"):
        check_feasible(tiny_solution)


def test_validity_checker_checker_rejects_nan_tolerance(tiny_solution):
    with pytest.raises(ValueError, match="finite"):
        check_feasible(tiny_solution, atol=float("nan"))

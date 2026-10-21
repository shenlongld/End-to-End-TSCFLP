import numpy as np
import pytest
import torch
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance

from ss_tscflp.baselines.sa import _Problem, solve_simulated_annealing
from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.problem import check_feasible, objective


@pytest.mark.parametrize("sizes", [(1, 5, 100), (3, 5, 8), (3, 10, 100)])
def test_sa_feasible_atomic_and_reproducible(sizes):
    td = TSCFLPGenerator(*sizes, seed=42)([1])
    first = solve_simulated_annealing(td, iterations=500, restarts=2)
    second = solve_simulated_annealing(td, iterations=500, restarts=2)
    assert first.feasible and check_feasible(first.solution).all_valid
    assert first.cost <= first.initial_cost + 1e-4
    assert first.cost == objective(first.solution).item() == second.cost
    assert torch.equal(
        first.solution["customer_station_assignment"],
        second.solution["customer_station_assignment"],
    )
    assert torch.equal(
        first.solution["station_warehouse_assignment"],
        second.solution["station_warehouse_assignment"],
    )
    assert first.proposals == 1000
    assert first.accepted_uphill_moves > 0


def test_sa_delta_matches_objective(tiny_instance):
    problem = _Problem(tiny_instance)
    rng = np.random.default_rng(15)
    state = problem.construct(rng, randomized=False)
    tested = 0
    for _ in range(1000):
        move = problem.propose(state, rng)
        if move is not None:
            before = objective(problem.td, numpy_state=(problem, state))
            problem.apply(state, move)
            assert objective(problem.td, numpy_state=(problem, state)) == pytest.approx(
                before + move.delta, abs=1e-8
            )
            tested += 1
    assert tested > 100


def test_sa_failure_has_no_fabricated_solution(tiny_instance):
    tiny_instance["warehouse_capacity"].zero_()
    result = solve_simulated_annealing(tiny_instance, iterations=10, restarts=2)
    assert not result.feasible
    assert result.solution is None and result.cost is None


def test_sa_checks_input_budget(tiny_instance):
    with pytest.raises(ValueError, match="budgets"):
        solve_simulated_annealing(tiny_instance, iterations=0)

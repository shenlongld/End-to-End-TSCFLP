# 保留原始实现中的科研回归测试。

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance
from test_problem import tiny_solution as tiny_solution

from ss_tscflp.baselines.sa import _Problem, solve_simulated_annealing
from ss_tscflp.env import TSCFLPEnv
from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.problem import (
    check_feasible,
    objective,
)


def env_step(env, td, action):
    td["action"] = torch.tensor(action, dtype=torch.int64)
    return env.step(td)["next"]


def test_env_phases_and_atomic_assignments(tiny_instance):
    env = TSCFLPEnv()
    td = env.reset(tiny_instance)
    assert td["phase"].shape == td["reward"].shape == td["done"].shape == (1,)
    assert td["customer_order"].tolist() == [[0, 1, 2]]
    for action in [0, 1, 0]:
        td = env_step(env, td, [action])
    assert td["phase"].item() == 1
    assert td["station_order"].tolist() == [[0, 1, -1]]
    assert td["num_active_stations"].item() == 2
    td = env_step(env, td, [0])
    td = env_step(env, td, [0])
    assert td["done"].item() and td["phase"].item() == 2
    assert check_feasible(td).all_valid
    assert td["station_warehouse_assignment"].tolist() == [[0, 0, -1]]
    assert not td["action_mask"].any()
    assert torch.equal(env.get_reward(td), -td["incremental_cost"])


def test_env_fixed_cost_only_once_and_hard_mask(tiny_instance):
    env = TSCFLPEnv()
    td = env_step(env, env.reset(tiny_instance), [0])
    assert td["incremental_cost"].item() == 24
    assert td["action_mask"].tolist() == [[False, True, True]]
    td = env_step(env, td, [1])
    td = env_step(env, td, [0])
    assert (
        td["incremental_cost"].item() == 58
    )  # 两个站点固定成本及顾客运输成本：G0 + G1 + 4*1 + 3*1 + 2*3。


def test_env_infeasible_deadend(tiny_instance):
    env = TSCFLPEnv(penalty=100000)
    tiny_instance["station_capacity"] = torch.tensor([[4.0, 3.0, 0.0]])
    td = env.reset(tiny_instance)
    td = env_step(env, td, [0])
    td = env_step(env, td, [1])
    assert td["infeasible"].item() and td["done"].item()
    assert td["reward"].item() == -100000
    assert td["customer_station_assignment"].tolist() == [[0, 1, -1]]
    after = env_step(env, td, [-1])
    assert torch.equal(after["incremental_cost"], td["incremental_cost"])
    assert torch.equal(after["reward"], td["reward"])


def test_env_warehouse_deadend_is_not_repaired(tiny_instance):
    env = TSCFLPEnv()
    tiny_instance["warehouse_capacity"] = torch.tensor([[5.0, 5.0]])
    td = env.reset(tiny_instance)
    for action in [0, 1, 0]:
        td = env_step(env, td, [action])
    assert td["done"].item() and td["infeasible"].item()
    assert (td["station_warehouse_assignment"] == -1).all()


def test_env_initial_deadend(tiny_instance):
    tiny_instance["station_capacity"].zero_()
    td = TSCFLPEnv().reset(tiny_instance)
    assert td["infeasible"].item() and td["done"].item()
    assert td["reward"].item() == -100000


def test_env_mixed_phase_batch_and_finished_rows(tiny_instance):
    env = TSCFLPEnv()
    td = env.reset(torch.cat([tiny_instance, tiny_instance], 0))
    for actions in [[2, 0], [2, 1], [0, 0]]:
        td = env_step(env, td, actions)
    # 单独推进其中一行，不应影响另一行的状态。
    first = env_step(env, td[:1], [0])
    mixed = torch.cat([first, td[1:]], 0)
    mixed = env_step(env, mixed, [0, 0])
    assert mixed["done"].tolist() == [True, False]
    mixed = env_step(env, mixed, [-1, 0])
    assert mixed["done"].all()
    assert check_feasible(mixed).all_valid


def test_env_reset_does_not_mutate_instance(tiny_instance):
    before = tiny_instance.clone()
    TSCFLPEnv().reset(tiny_instance)
    assert set(before.keys()) == set(tiny_instance.keys())
    assert all(torch.equal(before[key], tiny_instance[key]) for key in before.keys())


def test_env_action_mask_padded_when_warehouses_larger():
    td = TSCFLPEnv(generator_params={"n_w": 5, "n_s": 2, "n_c": 3}).reset(batch_size=[2])
    assert td["action_mask"].shape == (2, 5)
    assert not td["action_mask"][:, 2:].any()


def test_env_environment_is_rl4co_compatible():
    from rl4co.envs.common.base import RL4COEnvBase

    env = TSCFLPEnv()
    assert isinstance(env, RL4COEnvBase)
    assert len(env.dataset([8])) == 8


def test_env_mixed_customer_and_station_steps(tiny_instance):
    env = TSCFLPEnv()
    early = env.reset(tiny_instance)
    late = env.reset(tiny_instance)
    for action in [0, 1, 0]:
        late = env_step(env, late, [action])
    mixed: TensorDict = torch.cat([early, late.exclude("action")], 0)
    mixed = env_step(env, mixed, [0, 0])
    assert mixed["customer_step"].tolist() == [1, 3]
    assert mixed["station_step"].tolist() == [0, 1]


@pytest.mark.parametrize("sizes", [(1, 5, 100), (3, 5, 8), (3, 10, 100)])
def test_sa_sa_feasible_atomic_and_reproducible(sizes):
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


def test_sa_sa_delta_matches_complete_objective_after_moves(tiny_instance):
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


def test_sa_sa_failure_has_no_fabricated_solution(tiny_instance):
    tiny_instance["warehouse_capacity"].zero_()
    result = solve_simulated_annealing(tiny_instance, iterations=10, restarts=2)
    assert not result.feasible
    assert result.solution is None and result.cost is None


def test_sa_sa_checks_input_budget(tiny_instance):
    with pytest.raises(ValueError, match="budgets"):
        solve_simulated_annealing(tiny_instance, iterations=0)

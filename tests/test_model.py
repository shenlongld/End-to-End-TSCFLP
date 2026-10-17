# 保留原始实现中的科研回归测试。

import itertools
from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance
from test_problem import tiny_solution as tiny_solution

from ss_tscflp.common import set_seed
from ss_tscflp.dataset import load_checkpoint, save_checkpoint
from ss_tscflp.env import PathTSCFLPEnv, TSCFLPEnv
from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.infer import best_of_k
from ss_tscflp.metrics import cost_consistency_error
from ss_tscflp.model import (
    AssignmentAwareStationAggregation,
    CostAwareStaticEncoder,
    CustomerStationDecoder,
    ObjectiveRelationAdapter,
    PathTSCFLPPolicy,
    StationWarehouseDecoder,
    TSCFLPPolicy,
    center_feasible,
    load_path_policy,
    load_policy,
)
from ss_tscflp.problem import (
    check_feasible,
)
from ss_tscflp.train import PathTrainConfig, TrainConfig, grouped_rl_loss, run_training, train_path


@pytest.mark.parametrize("method", ["path", "two_stage"])
def test_npz_training_exact_resume_after_dataset_relocation(tmp_path, method):
    import shutil

    from ss_tscflp.dataset import save_instances

    folder = tmp_path / "data"
    for index, split in enumerate(("train", "val")):
        data = TSCFLPGenerator(2, 3, 4, seed=80 + index)([8])
        save_instances(data, folder / split / "instances.npz", metadata={"seed": 80 + index})
    common = dict(
        device="cpu",
        data_mode="npz",
        dataset_dir=str(folder),
        epochs=2,
        n_w=2,
        n_s=3,
        n_c=4,
        val_size=8,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        output=str(tmp_path / "full"),
    )
    if method == "path":
        config = PathTrainConfig(
            **common,
            train_instances=8,
            batch_instances=4,
            group_size=3,
            val_samples=3,
            sample_chunk=2,
        )
        train = train_path
    else:
        config = TrainConfig(**common, train_size=8, batch_size=4)
        train = run_training
    train(config)
    train(replace(config, epochs=1, output=str(tmp_path / "split")))
    relocated = tmp_path / "relocated data"
    shutil.copytree(folder, relocated)
    train(
        replace(
            config,
            output=str(tmp_path / "resumed"),
            dataset_dir=str(relocated),
            resume=str(tmp_path / "split/last.npz"),
        )
    )
    full = load_checkpoint(tmp_path / "full/last.npz")
    resumed = load_checkpoint(tmp_path / "resumed/last.npz")
    assert all(
        torch.equal(value, resumed["policy_state_dict"][key])
        for key, value in full["policy_state_dict"].items()
    )


def test_missing_checkpoint_or_data_fails_before_sampling(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Generator must not be called for a missing file")

    monkeypatch.setattr(TSCFLPGenerator, "__call__", unexpected)
    with pytest.raises(FileNotFoundError):
        train_path(PathTrainConfig(device="cpu", resume=str(tmp_path / "missing.npz")))
    with pytest.raises(ValueError, match="dataset_dir"):
        train_path(PathTrainConfig(device="cpu", data_mode="npz"))


def test_end_to_end_evaluation_outputs_and_failure_accounting(tmp_path, tiny_instance):
    import json

    from ss_tscflp.baselines.common import load_solutions
    from ss_tscflp.dataset import save_instances
    from ss_tscflp.evaluate import evaluate

    cfg = PathTrainConfig(
        device="cpu",
        epochs=1,
        train_instances=4,
        batch_instances=2,
        group_size=3,
        val_size=2,
        val_samples=2,
        sample_chunk=2,
        n_w=2,
        n_s=3,
        n_c=3,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        output=str(tmp_path / "trained"),
    )
    train_path(cfg)
    data = tiny_instance.repeat(2)
    data["warehouse_capacity"][1] = 0
    path = tmp_path / "test.npz"
    save_instances(data, path)
    report = evaluate(
        data=path,
        checkpoint=tmp_path / "trained/best.npz",
        output=tmp_path / "evaluation",
        device="cpu",
        samples=3,
        sample_chunk=2,
        batch_size=1,
        sa_iterations=10,
        sa_restarts=1,
    )
    assert len(report["results"]) == 6
    assert report["proven_optima"] == 0
    assert (tmp_path / "evaluation/results.csv").is_file()
    json.loads((tmp_path / "evaluation/report.json").read_text(encoding="utf-8"))
    for method, summary in report["methods"].items():
        assert summary["infeasible_rate"] == 0.5
        assert summary["gap_instances"] == 0
        solution, _ = load_solutions(data, tmp_path / "evaluation" / f"{method}.npz")
        assert check_feasible(solution).valid.tolist() == [True, False]


def test_encoder_encoder_shapes_and_optional_coordinates(tiny_instance):
    encoder = CostAwareStaticEncoder(32, 4, 1)
    for td in [tiny_instance, TSCFLPGenerator(seed=3)([8])]:
        out = encoder(td)
        for key, source in [
            ("H_C", "customer_demand"),
            ("H_S", "station_capacity"),
            ("H_W", "warehouse_capacity"),
        ]:
            assert out[key].shape == (*td[source].shape, 32)
            assert torch.isfinite(out[key]).all()


def test_encoder_encoder_responds_to_edge_costs(tiny_instance):
    encoder = CostAwareStaticEncoder(32, 4, 1)
    first = encoder(tiny_instance)
    changed = tiny_instance.clone()
    changed["cost_sc"][0, 0, 0] *= 10
    second = encoder(changed)
    assert not torch.allclose(first["H_S"], second["H_S"])


def test_encoder_encoder_station_permutation_equivariance(tiny_instance):
    encoder = CostAwareStaticEncoder(32, 4, 2)
    permutation = torch.tensor([2, 0, 1])
    changed = tiny_instance.clone()
    for key in ["station_capacity", "station_fixed_cost"]:
        changed[key] = changed[key][:, permutation]
    changed["cost_sc"] = changed["cost_sc"][:, permutation]
    changed["cost_ws"] = changed["cost_ws"][:, :, permutation]
    first, second = encoder(tiny_instance), encoder(changed)
    assert torch.allclose(first["H_S"][:, permutation], second["H_S"], atol=2e-6)
    assert torch.allclose(first["H_C"], second["H_C"], atol=2e-6)


def test_aggregation_demand_weighted_pool_and_inactive_zero(tiny_solution):
    tiny_solution["station_remaining_capacity"] = (
        tiny_solution["station_capacity"] - tiny_solution["station_load"]
    )
    hc = torch.arange(24).float().reshape(1, 3, 8).requires_grad_()
    embedding = TensorDict(
        {"H_C": hc, "H_S": torch.zeros(1, 3, 8), "H_W": torch.zeros(1, 2, 8)}, [1]
    )
    aggregator = AssignmentAwareStationAggregation(8)
    pooled = aggregator.weighted_pool(tiny_solution, embedding)
    assert torch.allclose(pooled[0, 0], (4 * hc[0, 0] + 2 * hc[0, 2]) / 6)
    assert torch.equal(pooled[0, 1], hc[0, 1])
    assert torch.equal(pooled[0, 2], torch.zeros(8))
    out = aggregator(tiny_solution, embedding)
    assert out["H_S_assigned"].shape == (1, 3, 8)
    out["H_S_assigned"].square().sum().backward()
    assert hc.grad is not None


def test_aggregation_aggregation_changes_with_assignments(tiny_solution):
    td = tiny_solution
    td["station_remaining_capacity"] = td["station_capacity"] - td["station_load"]
    embedding = TensorDict({"H_C": torch.randn(1, 3, 16), "H_S": torch.randn(1, 3, 16)}, [1])
    aggregator = AssignmentAwareStationAggregation(16)
    before = aggregator(td, embedding)["H_S_assigned"]
    td["customer_station_assignment"] = torch.tensor([[1, 0, 0]])
    td["station_load"] = torch.tensor([[5.0, 4.0, 0.0]])
    td["station_remaining_capacity"] = td["station_capacity"] - td["station_load"]
    assert not torch.allclose(before, aggregator(td, embedding)["H_S_assigned"])


def test_decoders_customer_decoder_hard_mask(tiny_instance):
    env = TSCFLPEnv()
    td = env.reset(tiny_instance)
    td["action"] = torch.tensor([0])
    td = env.step(td)["next"]
    embedding = CostAwareStaticEncoder(32, 4, 1)(td)
    logits = CustomerStationDecoder(32)(td, embedding)
    assert logits.shape == (1, 3)
    assert torch.isneginf(logits[0, 0])
    assert logits.softmax(-1)[0, 0].item() == 0
    assert torch.isfinite(logits[0, 1:]).all()


def test_decoders_station_decoder_hard_mask(tiny_instance):
    tiny_instance["warehouse_capacity"] = torch.tensor([[5.0, 9.0]])
    env = TSCFLPEnv()
    td = env.reset(tiny_instance)
    for action in [0, 1, 0]:
        td["action"] = torch.tensor([action])
        td = env.step(td)["next"]
    embedding = CostAwareStaticEncoder(32, 4, 1)(td)
    embedding = AssignmentAwareStationAggregation(32)(td, embedding)
    logits = StationWarehouseDecoder(32)(td, embedding)
    assert logits.shape == (1, 2)
    assert torch.isneginf(logits[0, 0])
    assert logits.softmax(-1).tolist() == [[0.0, 1.0]]


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_policy_policy_rollout_and_log_likelihood(decode_type):
    env = TSCFLPEnv(generator=TSCFLPGenerator(3, 5, 8, seed=77))
    td = env.reset(batch_size=[16])
    policy = TSCFLPPolicy(32, 4, 1)
    calls = {"encoder": 0, "aggregation": 0}

    def count(key):
        def hook(*args):
            calls[key] += 1

        return hook

    policy.encoder.register_forward_hook(count("encoder"))
    policy.aggregation.register_forward_hook(count("aggregation"))
    out = policy(td, env, decode_type=decode_type)
    state = out["state"]
    assert state["done"].all()
    assert calls == {"encoder": 1, "aggregation": 1}
    assert torch.isfinite(out["log_likelihood"]).all()
    assert torch.allclose(out["log_likelihood"], out["step_log_probs"].sum(-1))
    assert (out["actions"][~out["action_valid"]] == -1).all()
    assert (cost_consistency_error(state) < 1e-4).all()
    if (~state["infeasible"]).any():
        assert check_feasible(state[~state["infeasible"]]).all_valid
    assert (td["customer_station_assignment"] == -1).all()


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_policy_scale_3_10_100_rollout_keeps_actual_capacities(decode_type):
    env = TSCFLPEnv(generator=TSCFLPGenerator(3, 10, 100, seed=42))
    policy = TSCFLPPolicy(32, 4, 1)
    out = policy(env.reset(batch_size=[16]), env, decode_type=decode_type)
    state = out["state"]
    assert state["done"].all()
    successful = state[~state["infeasible"]]
    assert successful.batch_size[0] > 0
    assert check_feasible(successful).all_valid
    assert state["customer_station_assignment"].shape == (16, 100)
    assert state["station_warehouse_assignment"].shape == (16, 10)
    assert (cost_consistency_error(state) < 1e-4).all()
    assert torch.isfinite(out["log_likelihood"]).all()
    assert torch.allclose(out["log_likelihood"], out["step_log_probs"].sum(-1))
    assert (out["actions"][~out["action_valid"]] == -1).all()


def test_policy_policy_greedy_is_reproducible(tiny_instance):
    env, policy = TSCFLPEnv(), TSCFLPPolicy(32, 4, 1)
    first = policy(tiny_instance, env, decode_type="greedy")
    second = policy(tiny_instance, env, decode_type="greedy")
    assert torch.equal(first["actions"], second["actions"])
    assert torch.equal(first["reward"], second["reward"])


def test_policy_policy_all_deadends_without_nan(tiny_instance):
    tiny_instance["station_capacity"].zero_()
    env, policy = TSCFLPEnv(), TSCFLPPolicy(32, 4, 1)
    out = policy(tiny_instance, env)
    assert out["infeasible"].all()
    assert out["reward"].item() == -100000
    assert out["log_likelihood"].item() == 0
    assert out["actions"].shape == (1, 0)


def test_policy_policy_ignores_finished_rows(tiny_instance):
    failed = tiny_instance.clone()
    failed["station_capacity"].zero_()
    instance = torch.cat([failed, tiny_instance], 0)
    env, policy = TSCFLPEnv(), TSCFLPPolicy(32, 4, 1)
    out = policy(instance, env, decode_type="sampling")
    assert out["infeasible"].tolist() == [True, False]
    assert (out["actions"][0] == -1).all()
    assert out["log_likelihood"][0].item() == 0


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_policy_scale_1_5_100_single_warehouse_rollout(decode_type):
    env = TSCFLPEnv(generator=TSCFLPGenerator(1, 5, 100, seed=42))
    policy = TSCFLPPolicy(32, 4, 1)
    out = policy(env.reset(batch_size=[8]), env, decode_type=decode_type)
    state = out["state"]
    assert state["customer_station_assignment"].shape == (8, 100)
    assert state["station_warehouse_assignment"].shape == (8, 5)
    assert state["done"].all() and not state["infeasible"].any()
    assert check_feasible(state).all_valid
    assert (state["customer_step"] == 100).all()
    assert (state["station_warehouse_assignment"][state["station_open"]] == 0).all()
    assert (state["station_warehouse_assignment"][~state["station_open"]] == -1).all()
    assert torch.allclose(
        state["warehouse_load"].squeeze(-1), state["customer_demand"].double().sum(-1).float()
    )
    assert (cost_consistency_error(state) < 1e-4).all()
    phase_one = out["action_valid"] & (out["action_phases"] == 1)
    assert (out["actions"][phase_one] == 0).all()
    assert (out["step_log_probs"][phase_one] == 0).all()
    assert torch.isfinite(out["log_likelihood"]).all()


@pytest.mark.parametrize("sizes", [(3, 5, 8), (3, 10, 100)])
def test_backward_backward(sizes):
    env = TSCFLPEnv(generator=TSCFLPGenerator(*sizes, seed=5))
    policy = TSCFLPPolicy(32, 4, 1)
    td = env.reset(batch_size=[32])
    # 放宽测试容量，保证两个阶段的决策均具有随机性。
    td = env.reset(
        td.set("warehouse_capacity", td["warehouse_capacity"] + td["customer_demand"].sum(-1, True))
    )
    out = policy(td, env, decode_type="sampling")
    advantage = out["reward"] - out["reward"].mean()
    loss = -(advantage.detach() * out["log_likelihood"]).mean()
    loss.backward()
    for name in ["encoder", "customer_decoder", "station_decoder", "aggregation"]:
        params = list(getattr(policy, name).parameters())
        grads = [p.grad for p in params if p.grad is not None]
        assert grads, f"No gradients for {name}"
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert sum(grad.abs().sum().item() for grad in grads) > 0, name
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    before = policy.aggregation.mlp[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, policy.aggregation.mlp[0].weight)


def test_backward_single_warehouse_has_zero_routing_gradient_by_definition():
    env = TSCFLPEnv(generator=TSCFLPGenerator(1, 5, 100, seed=42))
    policy = TSCFLPPolicy(32, 4, 1)
    out = policy(env.reset(batch_size=[8]), env, decode_type="sampling")
    advantage = (out["reward"] - out["reward"].mean()).detach()
    loss = -(advantage * out["log_likelihood"]).mean()
    loss.backward()
    for module in [policy.encoder, policy.customer_decoder]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert sum(grad.abs().sum().item() for grad in gradients) > 0
    # 当 P(warehouse=0)=1 时，对数概率为零；
    # 对应梯度及聚合模块的路由梯度也必须为零。
    for module in [policy.station_decoder, policy.aggregation]:
        for parameter in module.parameters():
            assert parameter.grad is None or (parameter.grad == 0).all()


def test_path_frontier_every_tiny_feasible_solution_reachable_with_identical_cost(tiny_instance):
    # 枚举原问题的二元分配，逐一比较两种解码过程，而非依赖神经网络输出。
    path_env, old_env = PathTSCFLPEnv(), TSCFLPEnv()
    count = 0
    for cs in itertools.product(range(3), repeat=3):
        active = sorted(set(cs))
        for warehouses in itertools.product(range(2), repeat=len(active)):
            target = tiny_instance.clone()
            sw = [-1] * 3
            for station, warehouse in zip(active, warehouses, strict=True):
                sw[station] = warehouse
            target["customer_station_assignment"] = torch.tensor([cs])
            target["station_warehouse_assignment"] = torch.tensor([sw])
            if not check_feasible(target).all_valid:
                continue
            state, legacy = path_env.reset(tiny_instance), old_env.reset(tiny_instance)
            for _ in range(3):
                customer = int(state["current_customer"][0])
                station = cs[customer]
                action = station * 2 + sw[station]
                assert state["action_mask"][0, action]
                state["action"] = torch.tensor([action])
                state = path_env.step(state)["next"]
                legacy["action"] = torch.tensor([station])
                legacy = old_env.step(legacy)["next"]
            for _ in active:
                station = int(legacy["current_station"][0])
                legacy["action"] = torch.tensor([sw[station]])
                legacy = old_env.step(legacy)["next"]
            assert check_feasible(state).all_valid
            assert torch.equal(
                state["customer_station_assignment"], target["customer_station_assignment"]
            )
            assert torch.equal(
                state["station_warehouse_assignment"], target["station_warehouse_assignment"]
            )
            torch.testing.assert_close(state["_cost64"], legacy["_cost64"], rtol=0, atol=1e-10)
            torch.testing.assert_close(path_env.get_reward(state), old_env.get_reward(legacy))
            count += 1
    assert count > 50


def test_path_frontier_binding_hard_masks_and_no_state_mutation(tiny_instance):
    env = PathTSCFLPEnv()
    state = env.reset(tiny_instance)
    before = state.clone()
    state["action"] = torch.tensor([0])
    after = env.step(state)["next"]
    assert not after["action_mask"][0, 1]  # 站点 0 不能改分配到仓库 1。
    assert after["station_warehouse_assignment"][0, 0] == 0
    assert torch.equal(state["_station_load64"], before["_station_load64"])
    after["action"] = torch.tensor([1])
    with pytest.raises(ValueError, match="Masked"):
        env.step(after)


def test_path_frontier_path_avoids_unshippable_station_load(tiny_instance):
    td = tiny_instance.clone()
    td["customer_demand"][:] = 4
    td["station_capacity"][:] = 8
    td["warehouse_capacity"][:] = 6
    env = PathTSCFLPEnv()
    state = env.reset(td)
    state["action"] = torch.tensor([0])
    state = env.step(state)["next"]
    assert not state["action_mask"][0, :2].any()  # 原先仅检查站点的掩码会接受负载 8。
    assert state["action_mask"][0, 3]  # 改用另一站点和仓库时动作合法。


def test_path_frontier_full_network_backward_and_independent_validity():
    td = TSCFLPGenerator(n_w=3, n_s=5, n_c=12, seed=91)([3])
    policy = PathTSCFLPPolicy(32, 4, 1)
    out = policy(td, PathTSCFLPEnv(), num_samples=4)
    loss, _ = grouped_rl_loss(out["reward"].view(3, 4), out["log_likelihood"].view(3, 4))
    loss.backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in policy.parameters()) > 0
    good = ~out["infeasible"]
    assert good.any()
    assert check_feasible(out["state"][good]).all_valid


def test_path_frontier_failure_and_mixed_batch_are_finite(tiny_instance):
    td = tiny_instance.repeat(2)
    td["warehouse_capacity"][0] = 0
    policy = PathTSCFLPPolicy(16, 4, 1)
    out = policy(td, PathTSCFLPEnv(), decode_type="greedy")
    assert out["infeasible"][0]
    assert torch.isfinite(out["log_likelihood"]).all()
    assert torch.isfinite(out["entropy"]).all()
    assert out["log_likelihood"][0] == 0
    out["log_likelihood"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_path_frontier_all_immediately_infeasible_has_differentiable_zero(tiny_instance):
    tiny_instance["station_capacity"][:] = 0
    out = PathTSCFLPPolicy(16, 4, 1)(tiny_instance, PathTSCFLPEnv())
    assert out["actions"].shape == (1, 0)
    assert out["reward"].item() == -1e5
    out["log_likelihood"].sum().backward()


def test_path_frontier_currency_invariance_and_facility_equivariance(tiny_instance):
    policy = PathTSCFLPPolicy(32, 4, 1).eval()
    env = PathTSCFLPEnv()

    def logits(td):
        e = policy.encoder(td)
        return policy.decoder(env.reset(td), e, torch.zeros_like(e["hs"]))

    original = logits(tiny_instance)
    rescaled = tiny_instance.clone()
    for key in ("cost_sc", "cost_ws", "station_fixed_cost", "warehouse_fixed_cost"):
        rescaled[key] *= 17
    torch.testing.assert_close(logits(rescaled), original, rtol=1e-5, atol=1e-5)
    perm = tiny_instance.clone()
    sp, wp = torch.tensor([2, 0, 1]), torch.tensor([1, 0])
    for key in ("station_capacity", "station_fixed_cost"):
        perm[key] = perm[key][:, sp]
    for key in ("warehouse_capacity", "warehouse_fixed_cost"):
        perm[key] = perm[key][:, wp]
    perm["cost_sc"] = perm["cost_sc"][:, sp]
    perm["cost_ws"] = perm["cost_ws"][:, wp][:, :, sp]
    expected = original.reshape(1, 3, 2)[:, sp][:, :, wp].flatten(1)
    torch.testing.assert_close(logits(perm), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mixture", [0.0, 0.25, 1.0])
def test_path_frontier_exact_group_estimator_matches_enumerated_objective_gradient(mixture):
    # 通过精确有限求和验证 K 的缩放和基线实现，包括奖励并列情况。
    theta = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([1.0, 3.0, -4.0], dtype=torch.float64)
    tuples = torch.tensor(list(itertools.product(range(3), repeat=3)))
    logp = theta.log_softmax(-1)[tuples]
    rewards = values[tuples]
    probability = logp.sum(-1).exp()
    objective = (
        probability * ((1 - mixture) * rewards.mean(-1) + mixture * rewards.amax(-1))
    ).sum()
    true_gradient = torch.autograd.grad(-objective, theta, retain_graph=True)[0]
    estimate = theta.sum() * 0
    for row in range(tuples.shape[0]):
        loss, _ = grouped_rl_loss(
            rewards[row : row + 1], logp[row : row + 1], best_weight=mixture, reward_scale=1.0
        )
        estimate = estimate + probability[row].detach() * loss
    actual = torch.autograd.grad(estimate, theta)[0]
    torch.testing.assert_close(actual, true_gradient, rtol=1e-10, atol=1e-10)


def test_path_frontier_best_of_k_caches_encoder_and_flags_failures(tiny_instance):
    policy = PathTSCFLPPolicy(16, 4, 1)
    calls = []
    hook = policy.encoder.register_forward_hook(lambda *args: calls.append(1))
    result = best_of_k(policy, PathTSCFLPEnv(), tiny_instance, samples=5, sample_chunk=2)
    hook.remove()
    assert len(calls) == 1
    assert result["samples"] == 5
    assert check_feasible(result["state"]).all_valid
    tiny_instance["station_capacity"][:] = 0
    failed = best_of_k(policy, PathTSCFLPEnv(), tiny_instance, samples=3, sample_chunk=2)
    assert failed["infeasible"].all()
    assert failed["failed_candidates"].item() == 3


def test_path_frontier_training_checkpoint_and_exact_epoch_resume(tmp_path):
    cfg = PathTrainConfig(
        device="cpu",
        epochs=2,
        train_instances=4,
        batch_instances=2,
        group_size=3,
        val_size=2,
        val_samples=2,
        sample_chunk=2,
        n_w=2,
        n_s=3,
        n_c=5,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        best_warmup_epochs=0,
        output=str(tmp_path / "full"),
    )
    full = train_path(cfg)
    train_path(replace(cfg, epochs=1, output=str(tmp_path / "split")))
    resumed = train_path(
        replace(cfg, output=str(tmp_path / "split"), resume=str(tmp_path / "split" / "last.npz"))
    )
    a, _ = load_path_policy(tmp_path / "full" / "last.npz", device="cpu")
    b, _ = load_path_policy(tmp_path / "split" / "last.npz", device="cpu")
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key
    assert full["best_key"] == resumed["best_key"]
    assert full["total_training_trajectories"] == 24


def test_path_frontier_legacy_checkpoint_cannot_silently_load_as_path(tmp_path):
    path = tmp_path / "legacy.npz"
    save_checkpoint({"format_version": 1}, path)
    with pytest.raises(ValueError, match="format 2"):
        load_path_policy(path, device="cpu")


def test_research_feasible_centering_ignores_masked_nodes_and_common_offset():
    raw = torch.tensor([[10.0, 12.0, 10000.0]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(center_feasible(raw, mask)[mask], torch.tensor([-1.0, 1.0]))
    assert torch.equal(center_feasible(raw, mask), center_feasible(raw + 200, mask))


def test_research_centering_restores_gradient_after_common_offset_saturation():
    norms = {}
    for mode in ("baseline", "centered"):
        decoder = CustomerStationDecoder(32)
        decoder.configure_research(mode, False)
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.zero_()
            decoder.query[-1].bias.fill_(1)
            decoder.key.weight.copy_(torch.eye(32))
        nodes = torch.full((1, 3, 32), 100.0)
        nodes[0, :, 0] += torch.tensor([0.0, 0.02, 0.04])
        logits = decoder._logits(
            torch.zeros(1, 32),
            nodes,
            torch.zeros(1, 3),
            torch.zeros(1, 3, 3),
            torch.zeros(1, 3),
            torch.ones(1, 3, dtype=torch.bool),
        )
        (-logits.log_softmax(-1)[0, 0]).backward()
        norms[mode] = float(decoder.key.weight.grad.norm())
    assert norms["baseline"] == 0
    assert norms["centered"] > 0


def test_research_warehouse_marginal_feature_includes_load_and_one_time_fixed_cost():
    instance = TensorDict(
        {
            "customer_demand": torch.tensor([[4.0, 4.0]]),
            "station_capacity": torch.tensor([[10.0, 10.0]]),
            "warehouse_capacity": torch.tensor([[100.0, 100.0]]),
            "station_fixed_cost": torch.zeros(1, 2),
            "warehouse_fixed_cost": torch.tensor([[100.0, 5.0]]),
            "cost_sc": torch.tensor([[[0.0, 0.0], [0.1, 0.1]]]),
            "cost_ws": torch.tensor([[[0.1, 0.2], [1.0, 2.0]]]),
        },
        [1],
    )
    env = TSCFLPEnv()
    state = env.reset(instance)
    for action in (0, 1):
        state["action"] = torch.tensor([action])
        state = env.step(state)["next"]
    assert state["phase"].item() == 1 and state["current_station"].item() == 0
    features, marginal = ObjectiveRelationAdapter.features(state, 1)
    expected = torch.tensor([[100.4, 9.0]])
    assert torch.allclose(marginal, expected / expected.mean(-1, True))
    assert torch.isfinite(features).all()
    state["warehouse_open"][0, 1] = True
    _, marginal = ObjectiveRelationAdapter.features(state, 1)
    expected = torch.tensor([[100.4, 4.0]])
    assert torch.allclose(marginal, expected / expected.mean(-1, True))


@pytest.mark.parametrize(
    "options",
    [
        {"attention_mode": "centered"},
        {"attention_mode": "qknorm"},
        {"objective_features": True},
        {"objective_features": True, "objective_prior_strength": 0.0},
        {"attention_mode": "centered", "objective_features": True},
        {"attention_mode": "centered", "objective_features": True, "use_memory": True},
        {"attention_mode": "centered", "objective_features": True, "pooling_mode": "load_only"},
        {"attention_mode": "cost_only", "objective_features": True},
    ],
)
def test_research_research_architectures_preserve_hard_constraints_and_finite_gradients(options):
    env = TSCFLPEnv(generator=TSCFLPGenerator(3, 5, 12, seed=4))
    policy = TSCFLPPolicy(32, 4, 1, **options)
    out = policy(env.reset(batch_size=[8]), env, decode_type="sampling")
    assert torch.isfinite(out["log_likelihood"]).all()
    successful = out["state"][~out["infeasible"]]
    assert successful.batch_size[0] > 0
    assert check_feasible(successful).all_valid
    assert (cost_consistency_error(out["state"]) <= 1e-4).all()
    (out["reward"].detach() * out["log_likelihood"]).mean().backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_research_load_only_aggregation_is_invariant_to_equal_load_customer_composition():
    env = TSCFLPEnv(generator=TSCFLPGenerator(2, 2, 4, seed=7))
    state = env.reset(batch_size=[1])
    state["customer_demand"].fill_(2)
    state["station_load"].fill_(4)
    state["station_remaining_capacity"] = state["station_capacity"] - 4
    state["station_open"].fill_(True)
    state["customer_station_assignment"] = torch.tensor([[0, 0, 1, 1]])
    other = state.clone()
    other["customer_station_assignment"] = torch.tensor([[1, 0, 1, 0]])
    policy = TSCFLPPolicy(32, 4, 1, pooling_mode="load_only")
    embedding = policy.encoder(state)
    first = policy.aggregation(state, embedding)["H_S_assigned"]
    second = policy.aggregation(other, embedding)["H_S_assigned"]
    assert torch.equal(first, second)


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_research_zero_output_adapter_preserves_policy_and_can_learn(decode_type):
    env = TSCFLPEnv(generator=TSCFLPGenerator(3, 5, 12, seed=27))
    data = env.reset(batch_size=[8])
    baseline = TSCFLPPolicy(32, 4, 1)
    adapted = TSCFLPPolicy(32, 4, 1, objective_features=True, objective_prior_strength=0.0)
    adapted.load_state_dict(baseline.state_dict(), strict=False)
    set_seed(789, device="cpu")
    first = baseline(data, env, decode_type=decode_type)
    set_seed(789, device="cpu")
    second = adapted(data, env, decode_type=decode_type)
    for key in ("actions", "log_likelihood", "cost", "reward"):
        assert torch.equal(first[key], second[key])
    if decode_type == "sampling":
        (second["reward"].detach() * second["log_likelihood"]).mean().backward()
        weight = adapted.customer_decoder.objective_adapter.residual[-1].weight
        assert torch.isfinite(weight.grad).all() and weight.grad.norm() > 0


@pytest.mark.parametrize("which", [0, 1])
def test_downstream_ablation_local_features_ignore_future_costs_without_normalization_leak(
    tiny_instance, which
):
    env = TSCFLPEnv()
    state = env.reset(tiny_instance)
    if which == 1:
        for action in (0, 1, 2):
            state["action"] = torch.tensor([action])
            state = env.step(state)["next"]
        assert state["phase"].item() == 1 and state["current_station"].item() == 0
    perturbed = state.clone()
    if which == 0:
        perturbed["cost_ws"] *= 7
        perturbed["warehouse_capacity"] *= 3
    else:
        # 当前站点成本、局部容量及掩码保持固定。
        perturbed["cost_ws"][0, :, 1:] = torch.tensor([[40.0, 50.0], [0.1, 0.2]])
    local, price = ObjectiveRelationAdapter.features(state, which, False)
    local_other, price_other = ObjectiveRelationAdapter.features(perturbed, which, False)
    full, _ = ObjectiveRelationAdapter.features(state, which)
    full_other, _ = ObjectiveRelationAdapter.features(perturbed, which)
    assert torch.equal(local, local_other)
    assert torch.equal(price, price_other)
    assert not torch.equal(full, full_other)
    assert torch.equal(local[..., 3], torch.zeros_like(local[..., 3]))
    assert torch.isfinite(local).all()


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_downstream_ablation_ablated_branch_starts_with_exact_original_policy(decode_type):
    env = TSCFLPEnv(generator=TSCFLPGenerator(3, 5, 12, seed=28))
    data = env.reset(batch_size=[8])
    original = TSCFLPPolicy(32, 4, 1)
    local = TSCFLPPolicy(
        32,
        4,
        1,
        objective_features=True,
        objective_prior_strength=0.0,
        objective_downstream_features=False,
    )
    local.load_state_dict(original.state_dict(), strict=False)
    set_seed(456, device="cpu")
    first = original(data, env, decode_type=decode_type)
    set_seed(456, device="cpu")
    second = local(data, env, decode_type=decode_type)
    for key in ("actions", "cost", "reward", "log_likelihood"):
        assert torch.equal(first[key], second[key])


@pytest.mark.parametrize("invalid", [0, "false", None])
def test_downstream_ablation_downstream_option_rejects_ambiguous_values(invalid):
    with pytest.raises(ValueError):
        TSCFLPPolicy(objective_downstream_features=invalid)


@pytest.mark.parametrize("stages", ["customer", "warehouse"])
def test_downstream_stages_stage_selector_changes_only_the_other_residual(tiny_instance, stages):
    env = TSCFLPEnv()
    state = env.reset(tiny_instance)
    full = TSCFLPPolicy(32, 4, 1, objective_features=True, objective_prior_strength=0.0)
    selected = TSCFLPPolicy(
        32,
        4,
        1,
        objective_features=True,
        objective_prior_strength=0.0,
        objective_downstream_stages=stages,
    )
    with torch.no_grad():
        for decoder in (full.customer_decoder, full.station_decoder):
            residual = decoder.objective_adapter.residual
            residual[0].weight.fill_(0.1)
            residual[0].bias.zero_()
            residual[-1].weight.fill_(0.1)
            residual[-1].bias.zero_()
    selected.load_state_dict(full.state_dict(), strict=True)
    first = state.clone()
    for action in (0, 1, 2):
        state["action"] = torch.tensor([action])
        state = env.step(state)["next"]
    assert state["phase"].item() == 1
    for which, attr, td in ((0, "customer_decoder", first), (1, "station_decoder", state)):
        a = getattr(full, attr).objective_adapter(td, which)
        b = getattr(selected, attr).objective_adapter(td, which)
        kept = which == (0 if stages == "customer" else 1)
        assert torch.equal(a, b) if kept else not torch.equal(a, b)


def test_downstream_stages_legacy_disabled_switch_remains_disabled_in_both_stages():
    policy = TSCFLPPolicy(objective_features=True, objective_downstream_features=False)
    assert not policy.customer_decoder.objective_adapter.include_downstream
    assert not policy.station_decoder.objective_adapter.include_downstream


@pytest.mark.parametrize(
    "options",
    [
        {"objective_downstream_stages": "none"},
        {"objective_downstream_stages": None},
        {"objective_downstream_features": False, "objective_downstream_stages": "customer"},
    ],
)
def test_downstream_stages_invalid_stage_combinations_are_rejected(options):
    with pytest.raises(ValueError):
        TSCFLPPolicy(**options)


def eligible_reservation_warehouse_state(instance):
    data = instance.clone()
    data["warehouse_capacity"] = torch.tensor([[5.0, 9.0]])
    env = TSCFLPEnv()
    state = env.reset(data)
    for action in (0, 1, 2, 0):
        state["action"] = torch.tensor([action])
        state = env.step(state)["next"]
    # 首个完整站点占用仓库 0 的 5 单位容量中的 4 单位。
    # 未来站点负载为 2，因此只能分配到剩余容量为 9 的仓库 1。
    assert state["current_station"].item() == 1
    assert state["warehouse_remaining_capacity"].tolist() == [[1.0, 9.0]]
    return state


def test_eligible_reservation_reservation_excludes_destinations_that_cannot_hold_whole_station(
    tiny_instance,
):
    state = eligible_reservation_warehouse_state(tiny_instance)
    legacy, legacy_price = ObjectiveRelationAdapter.features(state, 1)
    eligible, eligible_price = ObjectiveRelationAdapter.features(
        state, 1, reservation_mode="eligible"
    )
    assert torch.allclose(eligible[..., 3], torch.tensor([[0.0, 2.0 / 9.0]]))
    assert legacy[0, 0, 3] > 0
    assert legacy[0, 1, 3] < eligible[0, 1, 3]
    other = [0, 1, 2, 4, 5, 6]
    assert torch.equal(legacy[..., other], eligible[..., other])
    assert torch.equal(legacy_price, eligible_price)


def test_eligible_reservation_no_eligible_destination_has_finite_zero_pressure_and_gradients(
    tiny_instance,
):
    state = eligible_reservation_warehouse_state(tiny_instance)
    state["warehouse_remaining_capacity"].fill_(1.0)
    costs = state["cost_ws"].detach().clone().requires_grad_(True)
    state["cost_ws"] = costs
    features, price = ObjectiveRelationAdapter.features(state, 1, reservation_mode="eligible")
    assert torch.isfinite(features).all() and torch.isfinite(price).all()
    assert torch.equal(features[..., 3], torch.zeros(1, 2))
    (features.sum() + price.sum()).backward()
    assert costs.grad is not None and torch.isfinite(costs.grad).all()


def test_eligible_reservation_all_destinations_eligible_recovers_original_reservation(
    tiny_instance,
):
    state = eligible_reservation_warehouse_state(tiny_instance)
    state["warehouse_remaining_capacity"].fill_(9.0)
    old, _ = ObjectiveRelationAdapter.features(state, 1)
    new, _ = ObjectiveRelationAdapter.features(state, 1, reservation_mode="eligible")
    assert torch.equal(old, new)
    # 只有站点 2 属于未来待分配站点；
    # 已分配站点和当前站点不能重复预留需求量。
    assert torch.allclose((new[..., 3] * 9.0).sum(-1), torch.tensor([2.0]))
    state["station_open"][0, 2] = False
    no_future, _ = ObjectiveRelationAdapter.features(state, 1, reservation_mode="eligible")
    assert torch.equal(no_future[..., 3], torch.zeros(1, 2))


def test_eligible_reservation_customer_branch_is_unchanged_even_after_training(tiny_instance):
    state = TSCFLPEnv().reset(tiny_instance)
    old = ObjectiveRelationAdapter(0.0)
    new = ObjectiveRelationAdapter(0.0, reservation_mode="eligible")
    with torch.no_grad():
        old.residual[-1].weight.fill_(0.2)
        old.residual[-1].bias.fill_(0.1)
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(old(state, 0), new(state, 0))
    later = eligible_reservation_warehouse_state(tiny_instance)
    assert not torch.equal(old(later, 1), new(later, 1))


@pytest.mark.parametrize(
    "options",
    [
        {"objective_reservation_mode": "unknown"},
        {"objective_reservation_mode": None},
        {"objective_reservation_mode": "eligible", "objective_features": False},
        {"objective_reservation_mode": "eligible", "objective_downstream_features": False},
        {"objective_reservation_mode": "eligible", "objective_downstream_stages": "customer"},
    ],
)
def test_eligible_reservation_invalid_or_unused_reservation_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        TSCFLPPolicy(**{"objective_features": True, **options})


def pooled_context_nonzero_adapter(mode):
    adapter = ObjectiveRelationAdapter(0.0, context_mode=mode)
    with torch.no_grad():
        adapter.residual[0].weight.zero_()
        adapter.residual[0].weight[0, 0] = 1.0
        adapter.residual[0].bias.zero_()
        adapter.residual[-1].weight.zero_()
        adapter.residual[-1].weight[0, 0] = 1.0
        adapter.residual[-1].bias.zero_()
    return adapter


@pytest.mark.parametrize("local_mode", ["local", "self_gate"])
def test_pooled_context_other_legal_candidates_condition_score_but_illegal_candidates_do_not(
    local_mode,
):
    features = torch.zeros(1, 3, 7)
    features[0, :, 0] = torch.tensor([1.0, 3.0, 100.0])
    mask = torch.tensor([[True, True, False]])
    local, pooled = (
        pooled_context_nonzero_adapter(local_mode),
        pooled_context_nonzero_adapter("pooled_gate"),
    )
    changed_legal = features.clone()
    changed_legal[0, 1, 0] = 5.0
    assert torch.equal(
        local.score_features(features, mask)[:, 0],
        local.score_features(changed_legal, mask)[:, 0],
    )
    assert not torch.equal(
        pooled.score_features(features, mask)[:, 0],
        pooled.score_features(changed_legal, mask)[:, 0],
    )
    changed_illegal = features.clone()
    changed_illegal[0, 2].fill_(1e6)
    assert torch.equal(
        pooled.score_features(features, mask)[:, :2],
        pooled.score_features(changed_illegal, mask)[:, :2],
    )


@pytest.mark.parametrize("mode", ["pooled_gate", "self_gate"])
def test_pooled_context_candidate_permutation_equivariance_and_empty_feasible_set_gradients(mode):
    adapter = pooled_context_nonzero_adapter(mode)
    features = torch.rand(3, 4, 7, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [False] * 4, [True, True, False, False]])
    permutation = torch.tensor([2, 0, 3, 1])
    first = adapter.score_features(features, mask)
    permuted = adapter.score_features(features[:, permutation], mask[:, permutation])
    assert torch.allclose(first[:, permutation], permuted, atol=1e-6)
    assert torch.isfinite(first).all()
    first.masked_fill(~mask, 0).sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert torch.equal(features.grad[1], torch.zeros_like(features.grad[1]))
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)


@pytest.mark.parametrize("mode", ["pooled_gate", "self_gate"])
def test_pooled_context_modes_have_identical_parameter_names_shapes_values_and_rng_consumption(
    mode,
):
    set_seed(954, device="cpu")
    local = ObjectiveRelationAdapter(0.0)
    after_local = torch.rand(3)
    set_seed(954, device="cpu")
    pooled = ObjectiveRelationAdapter(0.0, context_mode=mode)
    after_pooled = torch.rand(3)
    assert local.state_dict().keys() == pooled.state_dict().keys()
    for key, value in local.state_dict().items():
        assert torch.equal(value, pooled.state_dict()[key])
    assert torch.equal(after_local, after_pooled)


@pytest.mark.parametrize(
    "options",
    [
        {"objective_context_mode": None},
        {"objective_context_mode": "unknown"},
        {"objective_context_mode": "pooled_gate", "objective_features": False},
    ],
)
def test_pooled_context_invalid_context_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        TSCFLPPolicy(**{"objective_features": True, **options})


@pytest.mark.parametrize("sizes", [(2, 3, 4), (1, 5, 100)])
def test_scripts_rl4co_training_checkpoint_and_exact_resume(tmp_path, sizes):
    config = TrainConfig(
        device="cpu",
        epochs=2,
        train_size=16,
        val_size=16,
        batch_size=8,
        n_w=sizes[0],
        n_s=sizes[1],
        n_c=sizes[2],
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        seed=11,
        output=str(tmp_path / "continuous"),
    )
    uninterrupted = run_training(config)
    one_epoch = replace(config, epochs=1, output=str(tmp_path / "split"))
    run_training(one_epoch)
    resumed = run_training(
        replace(
            one_epoch,
            epochs=2,
            output=str(tmp_path / "resumed"),
            resume=str(tmp_path / "split" / "last.npz"),
        )
    )
    assert uninterrupted["history"] == resumed["history"]
    first = load_checkpoint(tmp_path / "continuous" / "last.npz")
    second = load_checkpoint(tmp_path / "resumed" / "last.npz")
    for key in first["policy_state_dict"]:
        assert torch.equal(first["policy_state_dict"][key], second["policy_state_dict"][key])
    policy, _ = load_policy(tmp_path / "resumed" / "best.npz", device="cpu")
    assert not policy.training
    assert all(row["gap_to_gurobi"] is None for row in resumed["history"])


def test_scripts_warm_start_keeps_best_actor_and_allows_larger_training_budget(tmp_path):
    first = TrainConfig(
        device="cpu",
        epochs=1,
        train_size=8,
        val_size=8,
        batch_size=4,
        n_w=2,
        n_s=3,
        n_c=4,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        seed=6,
        output=str(tmp_path / "first"),
    )
    previous = run_training(first)
    warmed = run_training(
        replace(
            first,
            train_size=16,
            output=str(tmp_path / "larger"),
            initialize_from=str(tmp_path / "first/last.npz"),
        )
    )
    assert warmed["initial_validation"]["reward"] == pytest.approx(previous["best_reward"])
    assert warmed["initialization"]["source_training_episodes"] == 8
    assert "fresh Adam" in warmed["initialization"]["optimizer"]
    last = load_checkpoint(tmp_path / "larger/last.npz")
    assert last["training_config"]["train_size"] == 16
    assert (tmp_path / "larger/best.npz").exists()
    assert warmed["best_reward"] >= warmed["initial_validation"]["reward"]

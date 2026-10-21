import itertools
from dataclasses import replace

import pytest
import torch
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance

from ss_tscflp.dataset import save_checkpoint
from ss_tscflp.env import Env
from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.infer import best_of_k
from ss_tscflp.model import Policy, load_policy
from ss_tscflp.problem import check_feasible, objective
from ss_tscflp.train import TrainConfig, group_loss, train


def test_all_tiny_solutions_reachable(tiny_instance):
    # 枚举问题的二元分配，独立检查路径可达性和完整四项成本。
    env = Env()
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
            state = env.reset(tiny_instance)
            for _ in range(3):
                customer = int(state["current_customer"][0])
                station = cs[customer]
                action = station * 2 + sw[station]
                assert state["action_mask"][0, action]
                state["action"] = torch.tensor([action])
                state = env.step(state)["next"]
            assert check_feasible(state).all_valid
            for key in ("customer_station_assignment", "station_warehouse_assignment"):
                assert torch.equal(state[key], target[key])
            torch.testing.assert_close(
                state["_cost64"], objective(target, dtype=torch.float64), rtol=0, atol=1e-10
            )
            torch.testing.assert_close(env.get_reward(state), -objective(target))
            count += 1
    assert count > 50


def test_binding_and_hard_masks(tiny_instance):
    env = Env()
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


def test_capacity_blocks_oversized_load(tiny_instance):
    td = tiny_instance.clone()
    td["customer_demand"][:] = 4
    td["station_capacity"][:] = 8
    td["warehouse_capacity"][:] = 6
    env = Env()
    state = env.reset(td)
    state["action"] = torch.tensor([0])
    state = env.step(state)["next"]
    assert not state["action_mask"][0, :2].any()  # 原先仅检查站点的掩码会接受负载 8。
    assert state["action_mask"][0, 3]  # 改用另一站点和仓库时动作合法。


def test_backward_and_feasibility():
    td = TSCFLPGenerator(n_w=3, n_s=5, n_c=12, seed=91)([3])
    policy = Policy(32, 4, 1)
    out = policy(td, Env(), num_samples=4)
    loss, _ = group_loss(out["reward"].view(3, 4), out["log_likelihood"].view(3, 4))
    loss.backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in policy.parameters()) > 0
    good = ~out["infeasible"]
    assert good.any()
    assert check_feasible(out["state"][good]).all_valid


def test_mixed_batch_failure(tiny_instance):
    td = tiny_instance.repeat(2)
    td["warehouse_capacity"][0] = 0
    policy = Policy(16, 4, 1)
    out = policy(td, Env(), decode_type="greedy")
    assert out["infeasible"][0]
    assert torch.isfinite(out["log_likelihood"]).all()
    assert torch.isfinite(out["entropy"]).all()
    assert out["log_likelihood"][0] == 0
    out["log_likelihood"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_all_failed_backward(tiny_instance):
    tiny_instance["station_capacity"][:] = 0
    out = Policy(16, 4, 1)(tiny_instance, Env())
    assert out["actions"].shape == (1, 0)
    assert out["reward"].item() == -1e5
    out["log_likelihood"].sum().backward()


def test_currency_and_permutations(tiny_instance):
    policy = Policy(32, 4, 1).eval()
    env = Env()

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
def test_group_loss_exact_gradient(mixture):
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
        loss, _ = group_loss(
            rewards[row : row + 1], logp[row : row + 1], best_weight=mixture, reward_scale=1.0
        )
        estimate = estimate + probability[row].detach() * loss
    actual = torch.autograd.grad(estimate, theta)[0]
    torch.testing.assert_close(actual, true_gradient, rtol=1e-10, atol=1e-10)


def test_sampling_reuses_encoder(tiny_instance):
    policy = Policy(16, 4, 1)
    calls = []
    hook = policy.encoder.register_forward_hook(lambda *args: calls.append(1))
    result = best_of_k(policy, Env(), tiny_instance, samples=5, sample_chunk=2)
    hook.remove()
    assert len(calls) == 1
    assert result["samples"] == 5
    assert check_feasible(result["state"]).all_valid
    tiny_instance["station_capacity"][:] = 0
    failed = best_of_k(policy, Env(), tiny_instance, samples=3, sample_chunk=2)
    assert failed["infeasible"].all()
    assert failed["failed_candidates"].item() == 3


def test_checkpoint_resume_exact(tmp_path):
    cfg = TrainConfig(
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
    full = train(cfg)
    train(replace(cfg, epochs=1, output=str(tmp_path / "split")))
    resumed = train(
        replace(cfg, output=str(tmp_path / "split"), resume=str(tmp_path / "split" / "last.npz"))
    )
    a, _ = load_policy(tmp_path / "full" / "last.npz", device="cpu")
    b, _ = load_policy(tmp_path / "split" / "last.npz", device="cpu")
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key
    assert full["best_key"] == resumed["best_key"]
    assert full["total_training_trajectories"] == 24


def test_checkpoint_rejects_old_format(tmp_path):
    path = tmp_path / "legacy.npz"
    save_checkpoint({"format_version": 1}, path)
    with pytest.raises(ValueError, match="format 2"):
        load_policy(path, device="cpu")

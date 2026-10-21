# 第145轮权重对应的耦合路径网络。

import math
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from .common import validate_device
from .dataset import load_checkpoint
from .env import Env
from .problem import objective


class FactorAttention(nn.Module):
    # 采用预归一化余弦注意力，并显式处理空邻域。

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.out = nn.Linear(dim, dim, bias=False)
        self.log_scale = nn.Parameter(torch.full((heads,), math.log(4.0)))
        self.ffnorm = nn.LayerNorm(dim)
        self.up, self.down = nn.Linear(dim, 4 * dim), nn.Linear(2 * dim, dim)

    def forward(self, query, nodes, bias=None, mask=None):
        b, n, dim = query.shape
        m = nodes.shape[1]
        q = self.q(self.qnorm(query)).view(b, n, self.heads, -1).transpose(1, 2)
        normalized = self.knorm(nodes)
        k = self.k(normalized).view(b, m, self.heads, -1).transpose(1, 2)
        v = self.v(normalized).view(b, m, self.heads, -1).transpose(1, 2)
        score = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        score = score * self.log_scale.exp().clamp(max=16)[None, :, None, None]
        if bias is not None:
            score = score + (bias[:, None] if bias.ndim == 3 else bias)
        if mask is not None:
            mask = mask[:, None]
            score = score.masked_fill(~mask, -torch.inf)
            score = torch.where(mask.any(-1, True), score, torch.zeros_like(score))
        weights = score.softmax(-1)
        if mask is not None:
            weights = weights.masked_fill(~mask, 0)
        h = query + self.out((weights @ v).transpose(1, 2).reshape(b, n, dim))
        gate, value = self.up(self.ffnorm(h)).chunk(2, -1)
        return h + self.down(F.silu(gate) * value)


class FactorEncoder(nn.Module):
    # 顾客、站点和仓库的类型化因子共用货币尺度，不使用节点编号嵌入。

    def __init__(self, dim: int, heads: int, layers: int):
        super().__init__()
        self.c = nn.Linear(4, dim)
        self.s = nn.Linear(6, dim)
        self.w = nn.Linear(5, dim)
        self.edge_sc = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.edge_ws = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, heads))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({k: FactorAttention(dim, heads) for k in ("cs", "sc", "ws", "sw")})
                for _ in range(layers)
            ]
        )

    def forward(self, td):
        d, qs, qw = (td[k] for k in ("customer_demand", "station_capacity", "warehouse_capacity"))
        cs, ws = td["cost_sc"], td["cost_ws"]
        fs, fw = td["station_fixed_cost"], td["warehouse_fixed_cost"]
        total, mean_d = d.sum(-1), d.mean(-1)
        # 统一货币尺度，保留固定成本与变动成本以及两个阶段之间的比例。
        unit = (cs.mean((1, 2)) + ws.mean((1, 2)) + (fs.sum(-1) + fw.sum(-1)) / total).clamp_min(
            1e-6
        )
        money = total * unit
        csn, wsn = cs / unit[:, None, None], ws / unit[:, None, None]
        hc = self.c(
            torch.stack([d / mean_d[:, None], d / total[:, None], csn.mean(1), csn.amin(1)], -1)
        )
        hs = self.s(
            torch.stack(
                [
                    qs / total[:, None],
                    fs / money[:, None],
                    fs / qs.clamp_min(1e-6) / unit[:, None],
                    csn.mean(-1),
                    wsn.mean(1),
                    wsn.amin(1),
                ],
                -1,
            )
        )
        hw = self.w(
            torch.stack(
                [
                    qw / total[:, None],
                    fw / money[:, None],
                    fw / qw.clamp_min(1e-6) / unit[:, None],
                    wsn.mean(-1),
                    wsn.amin(-1),
                ],
                -1,
            )
        )
        bsc = self.edge_sc(torch.stack([csn, csn * (d / mean_d[:, None])[:, None]], -1)).permute(
            0, 3, 1, 2
        )
        bws = self.edge_ws(torch.stack([wsn, wsn * (qs / total[:, None])[:, None]], -1)).permute(
            0, 3, 1, 2
        )
        for layer in self.layers:
            hc = layer["cs"](hc, hs, bsc.transpose(-1, -2))
            hs = layer["sc"](hs, hc, bsc)
            hw = layer["ws"](hw, hs, bws)
            hs = layer["sw"](hs, hw, bws.transpose(-1, -2))
        return TensorDict(
            {"hc": hc, "hs": hs, "hw": hw, "unit": unit, "step_scale": mean_d * unit},
            batch_size=td.batch_size,
        )


class PathDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, dynamic_context: bool = True):
        super().__init__()
        self.dynamic_context = dynamic_context
        self.station_dynamic = nn.Linear(5, dim)
        self.warehouse_dynamic = nn.Linear(4, dim)
        self.assigned = nn.Linear(dim, dim, bias=False)
        self.remaining = FactorAttention(dim, heads)
        self.station_to_warehouse = FactorAttention(dim, heads)
        self.warehouse_to_station = FactorAttention(dim, heads)
        self.query = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.LayerNorm(dim))
        self.path = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.features = nn.Sequential(nn.Linear(12, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        self.log_cost_strength = nn.Parameter(torch.tensor(0.0))
        self.log_context_cost = nn.Parameter(torch.tensor(0.0))
        # 神经网络残差初始较小，成本偏置仍可训练。
        nn.init.normal_(self.features[-1].weight, std=0.01)
        nn.init.zeros_(self.features[-1].bias)

    def forward(self, td, e, assigned_sum):
        d, qs, qw = (td[k] for k in ("customer_demand", "station_capacity", "warehouse_capacity"))
        b, ns = qs.shape
        nw = qw.shape[1]
        rows = torch.arange(b, device=d.device)
        current = td["current_customer"].clamp_min(0)
        mass, total = d[rows, current], d.sum(-1)
        rs, rw = td["station_remaining_capacity"], td["warehouse_remaining_capacity"]
        ls, lw = td["station_load"], td["warehouse_load"]
        opened_s, opened_w = td["station_open"], td["warehouse_open"]
        links = td["station_warehouse_assignment"]
        remaining = td["customer_station_assignment"] < 0
        remaining_mass = (remaining * d).sum(-1).clamp_min(1e-6)
        linked_capacity = rw.gather(1, links.clamp_min(0))
        reachable_w = torch.where(links >= 0, linked_capacity, rw.amax(-1, True))
        usable = torch.minimum(rs, reachable_w)
        sfeatures = torch.stack(
            [
                rs / qs.clamp_min(1e-6),
                ls / qs.clamp_min(1e-6),
                opened_s.float(),
                usable / remaining_mass[:, None],
                rs / remaining_mass[:, None],
            ],
            -1,
        )
        wfeatures = torch.stack(
            [
                rw / qw.clamp_min(1e-6),
                lw / qw.clamp_min(1e-6),
                opened_w.float(),
                rw / remaining_mass[:, None],
            ],
            -1,
        )
        hs = e["hs"] + self.station_dynamic(sfeatures)
        hs = hs + self.assigned(assigned_sum / ls[..., None].clamp_min(1e-6))
        hw = e["hw"] + self.warehouse_dynamic(wfeatures)
        link_mask = (links[:, :, None] < 0) | (
            links[:, :, None] == torch.arange(nw, device=d.device)[None, None]
        )
        edge_ws = td["cost_ws"].transpose(1, 2) / e["unit"][:, None, None]
        if self.dynamic_context:
            # 剩余顾客竞争站点的可用容量；空邻域的注意力为零，
            # 不通过虚构未屏蔽顾客来填充。
            feasible_future = remaining[:, None] & (d[:, None] <= usable[:, :, None])
            demand_bias = (d / total[:, None]).log()[:, None]
            cost_bias = -F.softplus(self.log_context_cost) * (
                td["cost_sc"] / e["unit"][:, None, None]
            )
            hs = self.remaining(hs, e["hc"], cost_bias + demand_bias, feasible_future)
            hw = self.station_to_warehouse(
                hw, hs, -edge_ws.transpose(1, 2), link_mask.transpose(1, 2)
            )
            hs = self.warehouse_to_station(hs, hw, -edge_ws, link_mask)
        demand_context = (e["hc"] * (remaining * d)[..., None]).sum(1)
        demand_context = demand_context / remaining_mass[:, None]
        query = self.query(
            torch.cat(
                [
                    e["hc"][rows, current],
                    demand_context,
                    (mass / remaining_mass)[:, None],
                    (td["customer_step"] / d.shape[1])[:, None],
                ],
                -1,
            )
        )
        sh = hs[:, :, None].expand(-1, -1, nw, -1)
        wh = hw[:, None].expand(-1, ns, -1, -1)
        paths = self.path(torch.cat([sh, wh, sh * wh], -1))
        score = (F.normalize(paths, dim=-1) * F.normalize(query, dim=-1)[:, None, None]).sum(-1)
        score = score * self.log_scale.exp().clamp(max=16)
        cs = td["cost_sc"].gather(2, current[:, None, None].expand(-1, ns, 1)).squeeze(2)
        transport_cs = (mass[:, None] * cs)[:, :, None].expand(-1, -1, nw)
        transport_ws = mass[:, None, None] * td["cost_ws"].transpose(1, 2)
        sf = (td["station_fixed_cost"] * ~opened_s)[:, :, None].expand(-1, -1, nw)
        wf = (td["warehouse_fixed_cost"] * ~opened_w)[:, None].expand(-1, ns, -1)
        increment = transport_cs + transport_ws + sf + wf
        scale = e["step_scale"][:, None, None]
        s_after = ((rs - mass[:, None]) / qs.clamp_min(1e-6))[:, :, None].expand_as(increment)
        w_after = ((rw - mass[:, None]) / qw.clamp_min(1e-6))[:, None].expand_as(increment)
        effective = torch.minimum(rs[:, :, None], rw[:, None]) / remaining_mass[:, None, None]
        features = torch.stack(
            [
                increment / scale,
                transport_cs / scale,
                transport_ws / scale,
                sf / scale,
                wf / scale,
                s_after,
                w_after,
                effective,
                opened_s[:, :, None].expand_as(increment).float(),
                opened_w[:, None].expand_as(increment).float(),
                (mass / remaining_mass)[:, None, None].expand_as(increment),
                (usable / remaining_mass[:, None])[:, :, None].expand_as(increment),
            ],
            -1,
        )
        score = score + self.features(features).squeeze(-1)
        score = score - F.softplus(self.log_cost_strength) * increment / scale
        return score.flatten(1).masked_fill(~td["action_mask"], -torch.inf)


class Policy(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=8, num_layers=2, temperature=1.0, dynamic_context=True
    ):
        super().__init__()
        if embed_dim < 8 or num_heads < 1 or embed_dim % num_heads or num_layers < 1:
            raise ValueError("Invalid embedding/head/layer configuration")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.config = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            temperature=temperature,
            dynamic_context=dynamic_context,
        )
        self.temperature = temperature
        self.encoder = FactorEncoder(embed_dim, num_heads, num_layers)
        self.decoder = PathDecoder(embed_dim, num_heads, dynamic_context)

    def forward(
        self,
        td,
        env: Env,
        decode_type="sampling",
        num_samples=1,
        return_actions=True,
        embeddings=None,
    ):
        if decode_type not in ("sampling", "greedy") or num_samples < 1:
            raise ValueError("Expected sampling/greedy and num_samples >= 1")
        if "phase" in td:
            raise ValueError("Path actor expects static instances, not a partial rollout")
        e = self.encoder(td) if embeddings is None else embeddings
        if num_samples > 1:
            # 样本排列：实例 0 的样本 0..K-1，实例 1 的样本 0..K-1，依次类推。
            e = e.repeat_interleave(num_samples, dim=0)
            td = td.repeat_interleave(num_samples, dim=0)
        state = env.reset(td)
        b, nc = state["customer_demand"].shape
        nw = state["warehouse_capacity"].shape[-1]
        log_likelihood = e["hc"].sum((1, 2)) * 0
        entropy_sum = log_likelihood.clone()
        assigned_sum = torch.zeros_like(e["hs"])
        actions, validity = [], []
        for _ in range(nc):
            running = ~state["done"]
            if not running.any():
                break
            logits = self.decoder(state, e, assigned_sum) / self.temperature
            # 已结束的行使用占位分布，其对统计量和梯度的贡献为零。
            logits = torch.where(running[:, None], logits, torch.zeros_like(logits))
            lp = logits.log_softmax(-1)
            chosen = (
                logits.argmax(-1)
                if decode_type == "greedy"
                else torch.multinomial(lp.exp(), 1).squeeze(-1)
            )
            log_likelihood = log_likelihood + lp.gather(1, chosen[:, None]).squeeze(1) * running
            safe_lp = lp.masked_fill(~torch.isfinite(lp), 0)
            entropy_sum = entropy_sum - (lp.exp() * safe_lp).sum(-1) * running
            current = state["current_customer"].clamp_min(0)
            rows = torch.arange(b, device=chosen.device)
            weighted = (
                e["hc"][rows, current]
                * (state["customer_demand"][rows, current] * running)[:, None]
            )
            index = (chosen // nw)[:, None, None].expand(-1, 1, e["hs"].shape[-1])
            assigned_sum = assigned_sum.scatter_add(1, index, weighted[:, None])
            action = chosen.masked_fill(~running, -1)
            if return_actions:
                actions.append(action)
                validity.append(running)
            state["action"] = action
            state = env.step(state)["next"]
        if not state["done"].all():
            raise RuntimeError("Path rollout exceeded customer count")
        out = {
            "reward": env.get_reward(state),
            "cost": objective(state),
            "log_likelihood": log_likelihood,
            "entropy": entropy_sum,
            "infeasible": state["infeasible"],
            "state": state,
        }
        if return_actions:
            out["actions"] = (
                torch.stack(actions, -1)
                if actions
                else torch.empty(b, 0, dtype=torch.long, device=log_likelihood.device)
            )
            out["action_valid"] = (
                torch.stack(validity, -1)
                if validity
                else torch.empty(b, 0, dtype=torch.bool, device=log_likelihood.device)
            )
        return out


def load_policy(path: str | Path, device: str):
    validate_device(device)
    payload = load_checkpoint(path)
    if payload.get("format_version") != 2 or payload.get("architecture") != "coupled_path_v1":
        raise ValueError("Expected coupled_path_v1 checkpoint format 2")
    policy = Policy(**payload["policy_config"]).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy, payload

# 保留原有两阶段、耦合路径神经网络及其消融配置。

import math
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from .common import validate_device
from .dataset import load_checkpoint
from .env import PathTSCFLPEnv, TSCFLPEnv
from .problem import objective


def normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, cost.ndim))
    return cost / cost.mean(dims, keepdim=True).clamp_min(1e-6)


class EdgeBiasedCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.q, self.k, self.v = (nn.Linear(embed_dim, embed_dim) for _ in range(3))
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, query: torch.Tensor, nodes: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, nq, dim = query.shape
        nk = nodes.shape[1]
        q = self.q(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(nodes).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(nodes).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + bias
        context = (score.softmax(-1) @ v).transpose(1, 2).reshape(b, nq, dim)
        hidden = self.norm1(query + self.out(context))
        return self.norm2(hidden + self.ff(hidden))


class CostAwareStaticEncoder(nn.Module):
    # 归一化特征，并沿顾客—站点和站点—仓库边交换信息。

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        if embed_dim < 8 or num_heads < 1 or embed_dim % num_heads or num_layers < 1:
            raise ValueError("embed_dim >= 8 must be divisible by num_heads; num_layers >= 1")
        self.embed_dim = embed_dim
        self.customer_projection = nn.Linear(5, embed_dim)
        self.station_projection = nn.Linear(6, embed_dim)
        self.warehouse_projection = nn.Linear(5, embed_dim)
        self.phi_sc = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, num_heads))
        self.phi_ws = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, num_heads))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        key: EdgeBiasedCrossAttention(embed_dim, num_heads)
                        for key in ("sc", "cs", "ws", "sw")
                    }
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _loc(td: TensorDict, key: str, like: torch.Tensor) -> torch.Tensor:
        return td.get(key, like.new_zeros(*like.shape, 2))

    def forward(self, td: TensorDict) -> TensorDict:
        demand, sc, wc = (
            td[key] for key in ("customer_demand", "station_capacity", "warehouse_capacity")
        )
        total = demand.sum(-1, keepdim=True).clamp_min(1e-6)
        csc, cws = normalize_cost(td["cost_sc"]), normalize_cost(td["cost_ws"])
        sf = normalize_cost(td["station_fixed_cost"])
        wf = normalize_cost(td["warehouse_fixed_cost"])
        hc = self.customer_projection(
            torch.cat(
                [
                    (demand / total)[..., None],
                    self._loc(td, "customer_loc", demand),
                    csc.mean(1)[..., None],
                    csc.amin(1)[..., None],
                ],
                -1,
            )
        )
        hs = self.station_projection(
            torch.cat(
                [
                    (sc / total)[..., None],
                    sf[..., None],
                    self._loc(td, "station_loc", sc),
                    csc.mean(-1)[..., None],
                    cws.mean(1)[..., None],
                ],
                -1,
            )
        )
        hw = self.warehouse_projection(
            torch.cat(
                [
                    (wc / total)[..., None],
                    wf[..., None],
                    self._loc(td, "warehouse_loc", wc),
                    cws.mean(-1)[..., None],
                ],
                -1,
            )
        )
        bias_sc = self.phi_sc(csc[..., None]).permute(0, 3, 1, 2)
        bias_ws = self.phi_ws(cws[..., None]).permute(0, 3, 1, 2)
        for layer in self.layers:
            new_c = layer["cs"](hc, hs, bias_sc.transpose(-2, -1))
            new_w = layer["ws"](hw, hs, bias_ws)
            new_s = 0.5 * (
                layer["sc"](hs, hc, bias_sc) + layer["sw"](hs, hw, bias_ws.transpose(-2, -1))
            )
            hc, hs, hw = new_c, new_s, new_w
        return TensorDict({"H_C": hc, "H_S": hs, "H_W": hw}, batch_size=td.batch_size)


class AssignmentAwareStationAggregation(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim + 3, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pooling_mode = "weighted"

    @staticmethod
    def weighted_pool(td: TensorDict, embeddings: TensorDict) -> torch.Tensor:
        hc, hs = embeddings["H_C"], embeddings["H_S"]
        assignment = td["customer_station_assignment"]
        valid = (assignment >= 0) & (assignment < hs.shape[1])
        index = assignment.clamp(0, hs.shape[1] - 1)
        weighted = hc * (td["customer_demand"] * valid)[..., None]
        sums = torch.zeros_like(hs).scatter_add(1, index[..., None].expand_as(hc), weighted)
        load = td["station_load"][..., None]
        return torch.where(load > 0, sums / load.clamp_min(1e-6), torch.zeros_like(sums))

    def forward(self, td: TensorDict, embeddings: TensorDict) -> TensorDict:
        load, cap = td["station_load"], td["station_capacity"].clamp_min(1e-6)
        features = torch.stack(
            [
                load / cap,
                td["station_remaining_capacity"] / cap,
                td["station_open"].float(),
            ],
            -1,
        )
        pooled = (
            self.weighted_pool(td, embeddings)
            if self.pooling_mode == "weighted"
            else torch.zeros_like(embeddings["H_S"])
        )
        assigned = self.mlp(torch.cat([embeddings["H_S"], pooled, features], -1))
        result = embeddings.clone(recurse=False)
        result["H_S_assigned"] = assigned
        return result


def center_feasible(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = scores.masked_fill(~mask, 0).sum(-1, True) / mask.sum(-1, True).clamp_min(1)
    return scores - mean


class ObjectiveRelationAdapter(nn.Module):
    # 阶段 0 使用下游成本下界，阶段 1 使用实际边际目标值。
    # 这些量仅作为策略特征，不额外加入奖励，也不执行软流量分配。
    # 阶段 0 尚未确定仓库分配，因此下游成本只是下界代理，而非实际运输成本。

    def __init__(
        self,
        prior_strength: float = 1.0,
        include_downstream: bool = True,
        reservation_mode: str = "cost",
        context_mode: str = "local",
    ) -> None:
        super().__init__()
        if not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and nonnegative")
        if not isinstance(include_downstream, bool):
            raise ValueError("include_downstream must be boolean")
        if reservation_mode not in ("cost", "eligible"):
            raise ValueError("reservation_mode must be cost or eligible")
        if context_mode not in ("local", "pooled_gate", "self_gate"):
            raise ValueError("context_mode must be local, pooled_gate or self_gate")
        self.prior_strength = prior_strength
        self.include_downstream = include_downstream
        self.reservation_mode = reservation_mode
        self.context_mode = context_mode
        self.log_strength = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.residual = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def features(
        td, which: int, include_downstream: bool = True, reservation_mode: str = "cost"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reservation_mode not in ("cost", "eligible"):
            raise ValueError("reservation_mode must be cost or eligible")
        rows = torch.arange(td.batch_size[0], device=td["customer_demand"].device)
        total = td["customer_demand"].sum(-1).clamp_min(1e-6)
        if which == 0:
            current = td["current_customer"].clamp_min(0)
            mass = td["customer_demand"][rows, current]
            n = td["station_capacity"].shape[-1]
            transport = mass[:, None] * td["cost_sc"].gather(
                -1, current[:, None, None].expand(-1, n, 1)
            ).squeeze(-1)
            opened = td["station_open"]
            activation = td["station_fixed_cost"] * (~opened)
            downstream = torch.zeros_like(transport)
            if include_downstream:
                proposed_load = td["station_load"] + mass[:, None]
                eligible = td["warehouse_capacity"][:, :, None] >= proposed_load[:, None, :]
                costs = td["cost_ws"].masked_fill(~eligible, float("inf"))
                lower = costs.amin(1)
                # 非法站点候选的神经网络特征也必须保持为有限值。
                fallback = td["cost_ws"].amax((1, 2))[:, None] + 10
                lower = torch.where(torch.isfinite(lower), lower, fallback)
                downstream = mass[:, None] * lower
            after = (td["station_remaining_capacity"] - mass[:, None]) / td[
                "station_capacity"
            ].clamp_min(1e-6)
            marginal = transport + activation + downstream
        else:
            current = td["current_station"].clamp_min(0)
            mass = td["station_load"][rows, current]
            n = td["warehouse_capacity"].shape[-1]
            transport = mass[:, None] * td["cost_ws"].gather(
                -1, current[:, None, None].expand(-1, n, 1)
            ).squeeze(-1)
            opened = td["warehouse_open"]
            activation = td["warehouse_fixed_cost"] * (~opened)
            downstream = torch.zeros_like(transport)
            if include_downstream:
                future = (td["station_warehouse_assignment"] < 0) & td["station_open"]
                future = future.clone()
                future[rows, current] = False
                future_load = td["station_load"] * future
                # 软预留特征：未来可能选择该仓库的期望需求量。
                reservation_logits = -td["cost_ws"] / td["cost_ws"].mean((1, 2), True).clamp_min(
                    1e-6
                )
                if reservation_mode == "eligible":
                    # 未来站点的负载不可拆分；本阶段剩余容量只会减少，
                    # 因此当前容纳不下该站点的仓库，后续也无法容纳。
                    eligible = (
                        td["warehouse_remaining_capacity"][:, :, None]
                        >= td["station_load"][:, None, :]
                    ) & future[:, None, :]
                    masked = reservation_logits.masked_fill(~eligible, -float("inf"))
                    # 没有可行目的地时不预留容量，也不进行软修复；
                    # 对应列的前向值和梯度均需保持为有限值。
                    masked = torch.where(eligible.any(1, True), masked, torch.zeros_like(masked))
                    routing = masked.softmax(1).masked_fill(~eligible, 0)
                else:
                    routing = reservation_logits.softmax(1)
                downstream = (routing * future_load[:, None, :]).sum(-1)
                downstream = downstream / td["warehouse_remaining_capacity"].clamp_min(1)
            after = (td["warehouse_remaining_capacity"] - mass[:, None]) / td[
                "warehouse_capacity"
            ].clamp_min(1e-6)
            marginal = transport + activation
        mask = td["action_mask"][:, :n]
        scale = marginal.masked_fill(~mask, 0).sum(-1, True) / mask.sum(-1, True).clamp_min(1)
        scale = scale.clamp_min(1e-6)
        features = torch.stack(
            [
                marginal / scale,
                transport / scale,
                activation / scale,
                downstream / scale if which == 0 else downstream,
                after,
                opened.float(),
                (mass / total)[:, None].expand(-1, n),
            ],
            -1,
        )
        return features, marginal / scale

    def score_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.context_mode == "local":
            return self.residual(features).squeeze(-1)
        projection, activation, output = self.residual
        projected = projection(features)
        hidden = activation(projected)
        if self.context_mode == "self_gate":
            # 使用匹配的激活对照，不引入其他候选的信息。
            gate = 1 + projected.tanh()
        else:
            # 对可行候选集合的摘要复用首层投影，不增加参数。
            # 非法候选不参与摘要计算。
            summary = features.masked_fill(~mask[..., None], 0).sum(1)
            summary = summary / mask.sum(1, True).clamp_min(1)
            gate = (1 + projection(summary).tanh())[:, None, :]
        return output(hidden * gate).squeeze(-1)

    def forward(self, td, which: int) -> torch.Tensor:
        features, price = self.features(td, which, self.include_downstream, self.reservation_mode)
        prior = self.prior_strength * F.softplus(self.log_strength) * price
        mask = td["action_mask"][:, : features.shape[1]]
        return self.score_features(features, mask) - prior


class SelectiveStateMemory(nn.Module):
    # 针对 TSCFLP 设计的轻量节点门控回合记忆模块，不声称复现 MiLoop。

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(5, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.gate = nn.Linear(embed_dim + 5, embed_dim)
        nn.init.constant_(self.gate.bias, -2.0)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, static, memory, features):
        proposed = self.norm(static + self.input(features))
        gate = self.gate(torch.cat([memory, features], -1)).sigmoid()
        return (1 - gate) * memory + gate * proposed


class _PointerDecoder(nn.Module):
    def __init__(self, embed_dim: int = 128, tanh_clipping: float = 10.0) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(2 * embed_dim + 3, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.dynamic = nn.Sequential(
            nn.Linear(3, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.psi = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.tanh_clipping = tanh_clipping
        self.attention_mode = "baseline"
        self.objective_adapter = None
        self.collect_diagnostics = False
        self.diagnostics = {"legal_scores": 0, "saturated": 0, "absolute_sum": 0.0}

    def configure_research(
        self,
        attention_mode: str,
        objective_features: bool,
        objective_prior_strength: float = 1.0,
        objective_downstream_features: bool = True,
        objective_reservation_mode: str = "cost",
        objective_context_mode: str = "local",
    ) -> None:
        if attention_mode not in ("baseline", "centered", "qknorm", "cost_only"):
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        self.attention_mode = attention_mode
        if attention_mode == "qknorm":
            self.log_attention_scale = nn.Parameter(torch.tensor(math.log(3.0)))
        if objective_features:
            self.objective_adapter = ObjectiveRelationAdapter(
                objective_prior_strength,
                objective_downstream_features,
                objective_reservation_mode,
                objective_context_mode,
            )

    def _logits(
        self,
        current: torch.Tensor,
        nodes: torch.Tensor,
        context: torch.Tensor,
        dynamic: torch.Tensor,
        costs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(torch.cat([current, nodes.mean(1), context], -1))
        keys = self.key(nodes + self.dynamic(dynamic))
        raw = (keys * query[:, None, :]).sum(-1) / math.sqrt(query.shape[-1])
        if self.collect_diagnostics:
            selected = raw.detach()[mask]
            self.diagnostics["legal_scores"] += selected.numel()
            self.diagnostics["saturated"] += int(((1 - selected.tanh().square()) == 0).sum())
            self.diagnostics["absolute_sum"] += float(selected.abs().sum())
        logits = raw
        if self.attention_mode == "centered":
            logits = center_feasible(raw, mask)
        elif self.attention_mode == "qknorm":
            logits = (F.normalize(keys, dim=-1) * F.normalize(query, dim=-1)[:, None]).sum(-1)
            logits = center_feasible(logits * self.log_attention_scale.exp().clamp(max=8), mask)
        elif self.attention_mode == "cost_only":
            logits = raw * 0
        logits = self.tanh_clipping * torch.tanh(logits) + self.psi(costs[..., None]).squeeze(-1)
        return logits.masked_fill(~mask, -float("inf"))


class CustomerStationDecoder(_PointerDecoder):
    def forward(self, td: TensorDict, embeddings: TensorDict) -> torch.Tensor:
        b, ns = td["station_capacity"].shape
        current = td["current_customer"].clamp_min(0)
        rows = torch.arange(b, device=current.device)
        cap = td["station_capacity"].clamp_min(1e-6)
        dynamic = torch.stack(
            [
                td["station_remaining_capacity"] / cap,
                td["station_load"] / cap,
                td["station_open"].float(),
            ],
            -1,
        )
        total = td["customer_demand"].sum(-1).clamp_min(1e-6)
        context = torch.stack(
            [
                td["customer_demand"][rows, current] / total,
                td["customer_step"].float() / td["customer_demand"].shape[-1],
                td["station_open"].float().mean(-1),
            ],
            -1,
        )
        costs = (
            normalize_cost(td["cost_sc"])
            .gather(-1, current[:, None, None].expand(-1, ns, 1))
            .squeeze(-1)
        )
        logits = self._logits(
            embeddings["H_C"][rows, current],
            embeddings["H_S"],
            context,
            dynamic,
            costs,
            td["action_mask"][:, :ns],
        )
        if self.objective_adapter is not None:
            logits = logits + self.objective_adapter(td, 0)
        return logits


class StationWarehouseDecoder(_PointerDecoder):
    def forward(self, td: TensorDict, embeddings: TensorDict) -> torch.Tensor:
        b, nw = td["warehouse_capacity"].shape
        current = td["current_station"].clamp_min(0)
        rows = torch.arange(b, device=current.device)
        cap = td["warehouse_capacity"].clamp_min(1e-6)
        dynamic = torch.stack(
            [
                td["warehouse_remaining_capacity"] / cap,
                td["warehouse_load"] / cap,
                td["warehouse_open"].float(),
            ],
            -1,
        )
        total = td["customer_demand"].sum(-1).clamp_min(1e-6)
        context = torch.stack(
            [
                td["station_load"][rows, current] / total,
                td["station_step"].float() / td["num_active_stations"].clamp_min(1),
                td["warehouse_open"].float().mean(-1),
            ],
            -1,
        )
        costs = (
            normalize_cost(td["cost_ws"])
            .gather(-1, current[:, None, None].expand(-1, nw, 1))
            .squeeze(-1)
        )
        logits = self._logits(
            embeddings["H_S_assigned"][rows, current],
            embeddings["H_W"],
            context,
            dynamic,
            costs,
            td["action_mask"][:, :nw],
        )
        if self.objective_adapter is not None:
            logits = logits + self.objective_adapter(td, 1)
        return logits


class TSCFLPPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        attention_mode: str = "baseline",
        objective_features: bool = False,
        objective_prior_strength: float = 1.0,
        pooling_mode: str = "weighted",
        use_memory: bool = False,
        objective_downstream_features: bool = True,
        objective_downstream_stages: str = "both",
        objective_reservation_mode: str = "cost",
        objective_context_mode: str = "local",
    ) -> None:
        super().__init__()
        if not 0 < temperature < float("inf") or not 0 < tanh_clipping < float("inf"):
            raise ValueError("temperature and tanh_clipping must be finite and positive")
        if not isinstance(objective_downstream_features, bool):
            raise ValueError("objective_downstream_features must be boolean")
        if objective_downstream_stages not in ("both", "customer", "warehouse"):
            raise ValueError("objective_downstream_stages must be both, customer or warehouse")
        if not objective_downstream_features and objective_downstream_stages != "both":
            raise ValueError("Disabled downstream features cannot have a selective stage")
        if objective_reservation_mode not in ("cost", "eligible"):
            raise ValueError("objective_reservation_mode must be cost or eligible")
        if objective_reservation_mode == "eligible" and (
            not objective_features
            or not objective_downstream_features
            or objective_downstream_stages == "customer"
        ):
            raise ValueError("eligible reservation requires warehouse downstream features")
        if objective_context_mode not in ("local", "pooled_gate", "self_gate"):
            raise ValueError("objective_context_mode must be local, pooled_gate or self_gate")
        if objective_context_mode != "local" and not objective_features:
            raise ValueError("Context gating requires objective features")
        self.config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "temperature": temperature,
            "tanh_clipping": tanh_clipping,
            "attention_mode": attention_mode,
            "objective_features": objective_features,
            "objective_prior_strength": objective_prior_strength,
            "objective_downstream_features": objective_downstream_features,
            "objective_downstream_stages": objective_downstream_stages,
            "objective_reservation_mode": objective_reservation_mode,
            "objective_context_mode": objective_context_mode,
            "pooling_mode": pooling_mode,
            "use_memory": use_memory,
        }
        self.temperature = temperature
        self.encoder = CostAwareStaticEncoder(embed_dim, num_heads, num_layers)
        self.aggregation = AssignmentAwareStationAggregation(embed_dim)
        self.customer_decoder = CustomerStationDecoder(embed_dim, tanh_clipping)
        self.station_decoder = StationWarehouseDecoder(embed_dim, tanh_clipping)
        if pooling_mode not in ("weighted", "load_only"):
            raise ValueError("pooling_mode must be weighted or load_only")
        self.aggregation.pooling_mode = pooling_mode
        self.customer_decoder.configure_research(
            attention_mode,
            objective_features,
            objective_prior_strength,
            objective_downstream_features and objective_downstream_stages in ("both", "customer"),
            objective_reservation_mode,
            objective_context_mode,
        )
        self.station_decoder.configure_research(
            attention_mode,
            objective_features,
            objective_prior_strength,
            objective_downstream_features and objective_downstream_stages in ("both", "warehouse"),
            objective_reservation_mode,
            objective_context_mode,
        )
        self.use_memory = use_memory
        if use_memory:
            self.station_memory = SelectiveStateMemory(embed_dim)
            self.warehouse_memory = SelectiveStateMemory(embed_dim)
        self.train_decode_type, self.val_decode_type, self.test_decode_type = (
            "sampling",
            "greedy",
            "greedy",
        )

    def forward(
        self,
        td: TensorDict,
        env: TSCFLPEnv,
        phase: str | None = None,
        decode_type: str | None = None,
        return_actions: bool = True,
        select_best: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs:
            raise ValueError(f"Unsupported policy arguments: {list(kwargs)}")
        if phase not in (None, "train", "val", "test"):
            raise ValueError("phase must be train, val or test")
        if decode_type is None:
            phase = phase or ("train" if self.training else "test")
            decode_type = getattr(self, f"{phase}_decode_type")
        if decode_type not in ("greedy", "sampling"):
            raise ValueError("Only greedy and sampling decoding are supported")
        state = td.clone() if "phase" in td else env.reset(td)
        embeddings = self.encoder(state)  # 每个回合只调用一次编码器。
        memory_s, memory_w = embeddings["H_S"], embeddings["H_W"]
        b, ns = state.batch_size[0], state["station_capacity"].shape[-1]
        device = state["phase"].device
        aggregated = torch.zeros(b, dtype=torch.bool, device=device)
        assigned_h = torch.zeros_like(embeddings["H_S"])
        likelihood = embeddings["H_C"].sum((1, 2)) * 0  # 立即进入死路时仍保留可求导的零值。
        actions, action_valid, action_phases, log_probs = [], [], [], []
        max_steps = state["customer_demand"].shape[-1] + ns

        for _ in range(max_steps + 1):
            new_rows = (
                ((state["customer_step"] == state["customer_demand"].shape[-1]) & ~aggregated)
                .nonzero()
                .flatten()
            )
            if new_rows.numel():
                aggregation_state, aggregation_embedding = (
                    (state, embeddings)
                    if new_rows.numel() == b
                    else (state[new_rows], embeddings[new_rows])
                )
                assigned = self.aggregation(aggregation_state, aggregation_embedding)[
                    "H_S_assigned"
                ]
                assigned_h = assigned_h.index_copy(0, new_rows, assigned)
                aggregated[new_rows] = True
            if state["done"].all():
                break
            step_embedding = embeddings.clone(recurse=False)
            step_embedding["H_S_assigned"] = assigned_h
            if self.use_memory:
                total = state["customer_demand"].sum(-1).clamp_min(1e-6)
                for which, static_key, capacity_key, remaining_key, load_key, open_key, memory in (
                    (
                        0,
                        "H_S",
                        "station_capacity",
                        "station_remaining_capacity",
                        "station_load",
                        "station_open",
                        memory_s,
                    ),
                    (
                        1,
                        "H_W",
                        "warehouse_capacity",
                        "warehouse_remaining_capacity",
                        "warehouse_load",
                        "warehouse_open",
                        memory_w,
                    ),
                ):
                    capacity = state[capacity_key].clamp_min(1e-6)
                    progress = (
                        state["customer_step"].float() / state["customer_demand"].shape[-1]
                        if which == 0
                        else state["station_step"].float()
                        / state["num_active_stations"].clamp_min(1)
                    )
                    features = torch.stack(
                        [
                            state[remaining_key] / capacity,
                            state[load_key] / capacity,
                            state[open_key].float(),
                            progress[:, None].expand_as(capacity),
                            (state[load_key] / total[:, None]),
                        ],
                        -1,
                    )
                    updater = self.station_memory if which == 0 else self.warehouse_memory
                    updated = updater(embeddings[static_key], memory, features)
                    active = ((state["phase"] == which) & ~state["done"])[:, None, None]
                    updated = torch.where(active, updated, memory)
                    step_embedding[static_key] = updated
                    if which == 0:
                        memory_s = updated
                    else:
                        memory_w = updated
            action = torch.full((b,), -1, dtype=torch.int64, device=device)
            step_lp = likelihood.new_zeros(b)
            running = ~state["done"]
            for which, decoder in ((0, self.customer_decoder), (1, self.station_decoder)):
                rows = ((state["phase"] == which) & running).nonzero().flatten()
                if not rows.numel():
                    continue
                # 当整个批次处于同一阶段时，避免每步复制全部顾客嵌入。
                # 混合阶段的行仍保留原索引和硬约束掩码。
                decoder_state, decoder_embedding = (
                    (state, step_embedding)
                    if rows.numel() == b
                    else (state[rows], step_embedding[rows])
                )
                logits = decoder(decoder_state, decoder_embedding) / self.temperature
                lp = logits.log_softmax(-1)
                chosen = (
                    logits.argmax(-1)
                    if decode_type == "greedy"
                    else torch.multinomial(lp.exp(), 1).squeeze(-1)
                )
                chosen_lp = lp.gather(-1, chosen[:, None]).squeeze(-1)
                action[rows] = chosen
                step_lp = step_lp.index_copy(0, rows, chosen_lp)
            likelihood = likelihood + step_lp
            actions.append(action)
            action_valid.append(running)
            action_phases.append(state["phase"].clone())
            log_probs.append(step_lp)
            state["action"] = action
            state = env.step(state)["next"]
        else:
            raise RuntimeError("Rollout exceeded the maximum number of atomic assignments")

        reward = env.get_reward(state)
        out = {
            "reward": reward,
            "cost": objective(state),
            "log_likelihood": likelihood,
            "state": state,
            "infeasible": state["infeasible"],
            "infeasible_rate": state["infeasible"].float(),
            "active_stations": state["station_open"].sum(-1).float(),
            "active_warehouses": state["warehouse_open"].sum(-1).float(),
        }
        if return_actions:
            out.update(
                {
                    "actions": torch.stack(actions, -1)
                    if actions
                    else torch.empty(b, 0, dtype=torch.int64, device=device),
                    "action_valid": torch.stack(action_valid, -1)
                    if actions
                    else torch.empty(b, 0, dtype=torch.bool, device=device),
                    "action_phases": torch.stack(action_phases, -1)
                    if actions
                    else torch.empty(b, 0, dtype=torch.int64, device=device),
                    "step_log_probs": torch.stack(log_probs, -1)
                    if actions
                    else likelihood[:, None][:, :0],
                }
            )
        return out


def load_policy(path: str | Path, device: str) -> tuple[TSCFLPPolicy, dict[str, Any]]:
    # 加载仅含张量和基础类型的检查点，不反序列化任意 Python 类。
    validate_device(device)
    payload = load_checkpoint(path)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported TSCFLP checkpoint format")
    policy = TSCFLPPolicy(**payload["policy_config"]).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy, payload


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


class PathFactorEncoder(nn.Module):
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


class ResidualPathDecoder(nn.Module):
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


class PathTSCFLPPolicy(nn.Module):
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
        self.encoder = PathFactorEncoder(embed_dim, num_heads, num_layers)
        self.decoder = ResidualPathDecoder(embed_dim, num_heads, dynamic_context)

    def forward(
        self,
        td,
        env: PathTSCFLPEnv,
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


def load_path_policy(path: str | Path, device: str):
    validate_device(device)
    payload = load_checkpoint(path)
    if payload.get("format_version") != 2 or payload.get("architecture") != "coupled_path_v1":
        raise ValueError("Expected coupled_path_v1 checkpoint format 2")
    policy = PathTSCFLPPolicy(**payload["policy_config"]).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy, payload

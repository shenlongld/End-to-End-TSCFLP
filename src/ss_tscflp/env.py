# 保留原有不可拆分解码过程，统一成本核算由 problem.py 提供。

import math
from typing import Any

import torch
from rl4co.envs.common.base import RL4COEnvBase
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from .generator import TSCFLPGenerator
from .problem import STATIC_KEYS, check_feasible, objective, step_increment, validate_instance


class TSCFLPEnv(RL4COEnvBase):
    # 阶段 0 为顾客分配站点；阶段 1 将每个启用站点整体分配给一个仓库。
    # 对外标量形状均为 [B]，张量数据类型保持 float32；内部负载和成本使用
    # float64 累加，以减少舍入漂移。采用 RL4CO 的 step(td)["next"] 快速接口。
    # 原生 TorchRL 采集器要求 done 和 reward 额外增加一维，因此此处不支持。

    name = "tscflp"

    def __init__(
        self,
        generator: TSCFLPGenerator | None = None,
        generator_params: dict[str, Any] | None = None,
        penalty: float = 1e5,
        **kwargs: Any,
    ) -> None:
        if not 0 < penalty < float("inf"):
            raise ValueError("penalty must be finite and positive")
        if kwargs.get("_torchrl_mode", False):
            raise ValueError("Use RL4CO fast mode to preserve the required [B] scalar schema")
        super().__init__(**kwargs)
        self.penalty = float(penalty)
        self._allow_generation = generator is not None or generator_params is not None
        params = dict(generator_params or {})
        params.setdefault("seed", self.rng.initial_seed())
        self.generator = generator if generator is not None else TSCFLPGenerator(**params)
        self._make_spec()

    def _set_seed(self, seed: int | None) -> None:
        self.rng = torch.Generator().manual_seed(0 if seed is None else seed)
        if hasattr(self, "generator"):
            self.generator.rng.manual_seed(self.rng.initial_seed())

    def _make_spec(self, td_params: TensorDict | None = None) -> None:
        nw, ns, nc = self.generator.n_w, self.generator.n_s, self.generator.n_c
        self.observation_spec = Composite(
            {
                "customer_demand": Unbounded(shape=(nc,), dtype=torch.float32),
                "station_capacity": Unbounded(shape=(ns,), dtype=torch.float32),
                "warehouse_capacity": Unbounded(shape=(nw,), dtype=torch.float32),
                "action_mask": Unbounded(shape=(max(ns, nw),), dtype=torch.bool),
            },
            device=self.device,
        )
        self.action_spec = Bounded(low=0, high=max(ns, nw) - 1, shape=(), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def reset(self, td: TensorDict | None = None, batch_size: Any = None) -> TensorDict:
        # EnvBase.reset 的后处理会将 done 从 [B] 扩展成 [B,1]。
        if td is None or td.is_empty():
            if not self._allow_generation:
                raise ValueError("reset requires instances or an explicitly supplied generator")
            td = self.generator(batch_size=[1] if batch_size is None else batch_size)
        self.to(td.device or td["customer_demand"].device)
        return self._reset(td, td.batch_size)

    def _reset(self, td: TensorDict, batch_size: Any = None) -> TensorDict:
        validate_instance(td)
        state = td.select(*(key for key in STATIC_KEYS if key in td)).clone()
        demand, sc, wc = (
            state[key]
            for key in (
                "customer_demand",
                "station_capacity",
                "warehouse_capacity",
            )
        )
        b, nc = demand.shape
        ns, nw, device = sc.shape[-1], wc.shape[-1], demand.device
        zero = torch.zeros(b, dtype=torch.int64, device=device)
        order = demand.argsort(dim=-1, descending=True, stable=True)
        state.update(
            {
                "phase": zero.clone(),
                "customer_order": order,
                "customer_step": zero.clone(),
                "current_customer": order[:, 0].clone(),
                "station_remaining_capacity": sc.clone(),
                "station_load": torch.zeros_like(sc),
                "station_open": torch.zeros_like(sc, dtype=torch.bool),
                "customer_station_assignment": torch.full(
                    (b, nc), -1, dtype=torch.int64, device=device
                ),
                "station_order": torch.full((b, ns), -1, dtype=torch.int64, device=device),
                "num_active_stations": zero.clone(),
                "station_step": zero.clone(),
                "current_station": torch.full_like(zero, -1),
                "warehouse_remaining_capacity": wc.clone(),
                "warehouse_load": torch.zeros_like(wc),
                "warehouse_open": torch.zeros_like(wc, dtype=torch.bool),
                "station_warehouse_assignment": torch.full(
                    (b, ns), -1, dtype=torch.int64, device=device
                ),
                "action_mask": torch.zeros(b, max(ns, nw), dtype=torch.bool, device=device),
                "incremental_cost": torch.zeros(b, dtype=torch.float32, device=device),
                "infeasible": torch.zeros(b, dtype=torch.bool, device=device),
                "done": torch.zeros(b, dtype=torch.bool, device=device),
                "reward": torch.zeros(b, dtype=torch.float32, device=device),
                "_cost64": torch.zeros(b, dtype=torch.float64, device=device),
                "_station_load64": torch.zeros_like(sc, dtype=torch.float64),
                "_warehouse_load64": torch.zeros_like(wc, dtype=torch.float64),
            }
        )
        return self._refresh(state)

    @torch.no_grad()
    def _step(self, td: TensorDict) -> TensorDict:
        state = td.clone()  # 保留策略反向传播所需张量，避免原地修改。
        action = state["action"]
        if action.dtype != torch.int64 or action.shape != state["phase"].shape:
            raise ValueError("action must be an int64 tensor with shape [B]")
        running = ~state["done"]
        safe = action.clamp(0, state["action_mask"].shape[-1] - 1)
        allowed = state["action_mask"].gather(-1, safe[:, None]).squeeze(-1)
        out_of_range = (action < 0) | (action >= state["action_mask"].shape[-1])
        if (running & (out_of_range | ~allowed)).any():
            raise ValueError("Action is masked or out of range for a running episode")

        rows = ((state["phase"] == 0) & running).nonzero().flatten()
        if rows.numel():
            customer, station = state["current_customer"][rows], action[rows]
            demand = state["customer_demand"][rows, customer].double()
            opened = ~state["station_open"][rows, station]
            state["_station_load64"][rows, station] += demand
            state["station_open"][rows, station] = True
            state["customer_station_assignment"][rows, customer] = station
            state["_cost64"][rows] += step_increment(
                state, rows, station=station, customer=customer, mass=demand, new_s=opened
            )
            state["customer_step"][rows] += 1

        # 本步只有在更新前已进入阶段 1 的行，才能执行阶段 1 的动作。
        rows = ((td["phase"] == 1) & running).nonzero().flatten()
        if rows.numel():
            station, warehouse = state["current_station"][rows], action[rows]
            load = state["_station_load64"][rows, station]
            opened = ~state["warehouse_open"][rows, warehouse]
            state["_warehouse_load64"][rows, warehouse] += load
            state["warehouse_open"][rows, warehouse] = True
            state["station_warehouse_assignment"][rows, station] = warehouse
            state["_cost64"][rows] += step_increment(
                state, rows, station=station, warehouse=warehouse, mass=load, new_w=opened
            )
            state["station_step"][rows] += 1

        state["station_load"] = state["_station_load64"].float()
        state["warehouse_load"] = state["_warehouse_load64"].float()
        state["station_remaining_capacity"] = (
            state["station_capacity"].double() - state["_station_load64"]
        ).float()
        state["warehouse_remaining_capacity"] = (
            state["warehouse_capacity"].double() - state["_warehouse_load64"]
        ).float()
        state["incremental_cost"] = state["_cost64"].float()
        transition = (state["phase"] == 0) & (
            state["customer_step"] == state["customer_demand"].shape[-1]
        )
        rows = transition.nonzero().flatten()
        if rows.numel():
            order = state["_station_load64"][rows].argsort(dim=-1, descending=True, stable=True)
            active = state["_station_load64"][rows].gather(-1, order) > 0
            state["station_order"][rows] = order.masked_fill(~active, -1)
            state["num_active_stations"][rows] = active.sum(-1)
            state["phase"][rows] = 1
        finished = (state["phase"] == 1) & (state["station_step"] >= state["num_active_stations"])
        state["done"][finished] = True
        state["phase"][finished] = 2
        state["reward"][finished] = -objective(state[finished])
        return self._refresh(state)

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        b, ns, nw = (
            td.batch_size[0],
            td["station_capacity"].shape[-1],
            td["warehouse_capacity"].shape[-1],
        )
        mask = torch.zeros(b, max(ns, nw), dtype=torch.bool, device=td["phase"].device)
        customer = td["current_customer"].clamp_min(0)
        demand = td["customer_demand"].gather(-1, customer[:, None]).double()
        station = td["current_station"].clamp_min(0)
        load = td["_station_load64"].gather(-1, station[:, None])
        mask[:, :ns] |= (
            (td["station_capacity"].double() - td["_station_load64"] >= demand)
            & (td["phase"] == 0)[:, None]
            & ~td["done"][:, None]
        )
        mask[:, :nw] |= (
            (td["warehouse_capacity"].double() - td["_warehouse_load64"] >= load)
            & (td["phase"] == 1)[:, None]
            & ~td["done"][:, None]
        )
        return mask

    def _refresh(self, state: TensorDict) -> TensorDict:
        nc, ns = state["customer_demand"].shape[-1], state["station_capacity"].shape[-1]
        customer = (
            state["customer_order"]
            .gather(-1, state["customer_step"].clamp_max(nc - 1)[:, None])
            .squeeze(-1)
        )
        station = (
            state["station_order"]
            .gather(-1, state["station_step"].clamp_max(ns - 1)[:, None])
            .squeeze(-1)
        )
        state["current_customer"] = customer.masked_fill(state["phase"] != 0, -1)
        state["current_station"] = station.masked_fill(state["phase"] != 1, -1)
        state["action_mask"] = self.get_action_mask(state)
        stuck = ~state["done"] & ~state["action_mask"].any(-1)
        state["infeasible"][stuck] = True
        state["done"][stuck] = True
        state["phase"][stuck] = 2
        state["current_customer"][stuck] = -1
        state["current_station"][stuck] = -1
        state["reward"][stuck] = -self.penalty
        return state

    def _get_reward(self, td: TensorDict, actions: torch.Tensor | None = None) -> torch.Tensor:
        if not td["done"].all():
            raise ValueError("Episodic reward requires all episodes to finish")
        return td["reward"]

    def get_reward(
        self,
        td: TensorDict,
        actions: torch.Tensor | None = None,
        check_solution: bool | None = None,
    ) -> torch.Tensor:
        if self.check_solution if check_solution is None else check_solution:
            self.check_solution_validity(td, actions)
        return self._get_reward(td, actions)

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor | None = None) -> None:
        successful = td["done"] & ~td["infeasible"]
        if successful.any():
            check_feasible(td[successful], raise_on_error=True)


class PathTSCFLPEnv:
    name = "tscflp_path"

    def __init__(self, generator=None, penalty: float = 1e5):
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("penalty must be finite and positive")
        self.generator = generator
        self.penalty = float(penalty)

    def reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:
        if td is None:
            if self.generator is None:
                raise ValueError("reset requires instances or an explicitly supplied generator")
            td = self.generator([1] if batch_size is None else batch_size)
        validate_instance(td)
        state = td.select(*(k for k in STATIC_KEYS if k in td)).clone()
        d, s, w = (state[k] for k in ("customer_demand", "station_capacity", "warehouse_capacity"))
        b, nc = d.shape
        device = d.device
        z = torch.zeros(b, dtype=torch.long, device=device)
        state.update(
            {
                "customer_order": d.argsort(dim=-1, descending=True, stable=True),
                "customer_step": z.clone(),
                "current_customer": z.clone(),
                "phase": z.clone(),
                "station_open": torch.zeros_like(s, dtype=torch.bool),
                "warehouse_open": torch.zeros_like(w, dtype=torch.bool),
                "customer_station_assignment": torch.full_like(d, -1, dtype=torch.long),
                "station_warehouse_assignment": torch.full_like(s, -1, dtype=torch.long),
                "_station_load64": torch.zeros_like(s, dtype=torch.float64),
                "_warehouse_load64": torch.zeros_like(w, dtype=torch.float64),
                "_cost64": torch.zeros(b, dtype=torch.float64, device=device),
                "done": torch.zeros(b, dtype=torch.bool, device=device),
                "infeasible": torch.zeros(b, dtype=torch.bool, device=device),
                "reward": torch.zeros(b, device=device),
            }
        )
        return self._refresh(state)

    def get_action_mask(self, state: TensorDict) -> torch.Tensor:
        # 掩码形状为 [B,S,W]，按 station * W + warehouse 展平；True 表示动作合法。
        nw = state["warehouse_capacity"].shape[-1]
        current = state["current_customer"].clamp_min(0)
        mass = state["customer_demand"].gather(1, current[:, None]).double()
        rem_s = state["station_capacity"].double() - state["_station_load64"]
        rem_w = state["warehouse_capacity"].double() - state["_warehouse_load64"]
        links = state["station_warehouse_assignment"]
        warehouses = torch.arange(nw, device=mass.device)[None, None, :]
        allowed_link = (links[:, :, None] < 0) | (links[:, :, None] == warehouses)
        return (
            (rem_s[:, :, None] >= mass[:, :, None])
            & (rem_w[:, None, :] >= mass[:, :, None])
            & allowed_link
            & ~state["done"][:, None, None]
        ).flatten(1)

    def _refresh(self, state: TensorDict) -> TensorDict:
        nc = state["customer_demand"].shape[-1]
        state["station_load"] = state["_station_load64"].float()
        state["warehouse_load"] = state["_warehouse_load64"].float()
        for kind in ("station", "warehouse"):
            state[f"{kind}_remaining_capacity"] = (
                state[f"{kind}_capacity"].double() - state[f"_{kind}_load64"]
            ).float()
        state["incremental_cost"] = state["_cost64"].float()
        finished = state["customer_step"] >= nc
        state["done"] |= finished
        state["current_customer"] = (
            state["customer_order"]
            .gather(1, state["customer_step"].clamp_max(nc - 1)[:, None])
            .squeeze(1)
        )
        state["action_mask"] = self.get_action_mask(state)
        stuck = ~state["done"] & ~state["action_mask"].any(-1)
        state["infeasible"] |= stuck
        state["done"] |= stuck
        state["phase"] = torch.where(state["done"], 2, 0)
        state["current_customer"] = state["current_customer"].masked_fill(state["done"], -1)
        state["reward"] = torch.where(
            state["infeasible"],
            -self.penalty,
            torch.where(finished, -objective(state), 0.0),
        )
        return state

    @torch.no_grad()
    def step(self, td: TensorDict) -> dict[str, TensorDict]:
        # 静态输入可以共享；自动求导保存的动态张量在后续步骤中必须保持不变。
        state = TensorDict(
            {k: v if k in STATIC_KEYS else v.clone() for k, v in td.items()},
            batch_size=td.batch_size,
        )
        action = state["action"]
        if action.dtype != torch.long or action.shape != state["done"].shape:
            raise ValueError("action must be int64 [B]")
        running = ~state["done"]
        safe = action.clamp(0, state["action_mask"].shape[1] - 1)
        legal = state["action_mask"].gather(1, safe[:, None]).squeeze(1)
        invalid = (action < 0) | (action >= state["action_mask"].shape[1]) | ~legal
        if (running & invalid).any():
            raise ValueError("Masked or out-of-range path action")
        rows = running.nonzero().flatten()
        nw = state["warehouse_capacity"].shape[1]
        station, warehouse = action[rows] // nw, action[rows] % nw
        customer = state["current_customer"][rows]
        mass = state["customer_demand"][rows, customer].double()
        new_s = ~state["station_open"][rows, station]
        new_w = ~state["warehouse_open"][rows, warehouse]
        state["_cost64"][rows] += step_increment(
            state,
            rows,
            station=station,
            warehouse=warehouse,
            customer=customer,
            mass=mass,
            new_s=new_s,
            new_w=new_w,
        )
        state["_station_load64"][rows, station] += mass
        state["_warehouse_load64"][rows, warehouse] += mass
        state["customer_station_assignment"][rows, customer] = station
        state["station_warehouse_assignment"][rows, station] = warehouse
        state["station_open"][rows, station] = True
        state["warehouse_open"][rows, warehouse] = True
        state["customer_step"][rows] += 1
        return {"next": self._refresh(state)}

    def get_reward(self, state: TensorDict, check_solution: bool = True) -> torch.Tensor:
        if not state["done"].all():
            raise ValueError("Reward requires completed episodes")
        success = ~state["infeasible"]
        if check_solution and success.any():
            check_feasible(state[success], raise_on_error=True)
        return state["reward"]

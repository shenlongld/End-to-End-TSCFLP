# 客户逐步选择站点和仓库，执行单源分配与两级硬容量约束。

import math

import torch
from tensordict import TensorDict

from .problem import STATIC_KEYS, check_feasible, objective, step_increment, validate_instance


class Env:
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

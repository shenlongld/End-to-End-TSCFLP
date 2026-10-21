# SS-TSCFLP 的统一数据结构、目标函数、可行性检查及最优间隙计算。

import math
from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

STATIC_KEYS = (
    "warehouse_capacity",
    "station_capacity",
    "customer_demand",
    "warehouse_fixed_cost",
    "station_fixed_cost",
    "cost_ws",
    "cost_sc",
    "warehouse_loc",
    "station_loc",
    "customer_loc",
)


def validate_instance(td: TensorDict) -> None:
    # 校验外部输入的数据结构，不预先假定该实例存在可行解。
    if len(td.batch_size) != 1 or td.batch_size[0] < 1:
        raise ValueError("Instance must have leading batch shape [B], B >= 1")
    required = STATIC_KEYS[:7]
    if any(key not in td for key in required):
        raise ValueError(f"Missing instance fields: {[key for key in required if key not in td]}")
    b = td.batch_size[0]
    for key in required[:5]:
        if td[key].ndim != 2 or td[key].shape[0] != b or td[key].shape[-1] < 1:
            raise ValueError(f"Invalid shape for {key}")
    nw, ns, nc = (td[key].shape[-1] for key in required[:3])
    expected = {
        "warehouse_fixed_cost": (b, nw),
        "station_fixed_cost": (b, ns),
        "cost_ws": (b, nw, ns),
        "cost_sc": (b, ns, nc),
        "warehouse_loc": (b, nw, 2),
        "station_loc": (b, ns, 2),
        "customer_loc": (b, nc, 2),
    }
    for key, shape in expected.items():
        if key in td and tuple(td[key].shape) != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in STATIC_KEYS:
        if key in td:
            value = td[key]
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{key} must contain finite float32 values")
            if (value < 0).any():
                raise ValueError(f"{key} cannot be negative")
    if (td["customer_demand"] <= 0).any():
        raise ValueError("Customer demands must be strictly positive")


@dataclass(frozen=True)
class ValidityReport:
    valid: torch.Tensor
    violations: dict[str, torch.Tensor]
    station_load: torch.Tensor
    warehouse_load: torch.Tensor

    @property
    def all_valid(self) -> bool:
        return bool(self.valid.all())


def assignment_matrices(td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回二元分配矩阵 z[B,S,C] 和 a[B,W,S]；索引 -1 表示尚未分配。
    ns = td["station_capacity"].shape[-1]
    nw = td["warehouse_capacity"].shape[-1]
    cs, sw = td["customer_station_assignment"], td["station_warehouse_assignment"]
    z = cs[:, None, :] == torch.arange(ns, device=cs.device)[None, :, None]
    a = sw[:, None, :] == torch.arange(nw, device=sw.device)[None, :, None]
    return z, a


def check_feasible(
    td: TensorDict,
    solution=None,
    atol: float = 1e-5,
    raise_on_error: bool = False,
) -> ValidityReport:
    # 检查分配完整性、单源约束、两级容量，以及设施启用与正负载的等价关系。
    # 返回逐实例的约束违反情况；数据结构错误时抛出 ValueError。
    # 结构正确但不可行或未完成的分配返回 valid=False。
    td = as_tensordict(td, solution)
    validate_instance(td)
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    b, nc = td["customer_demand"].shape
    ns = td["station_capacity"].shape[-1]
    for key, shape in (
        ("customer_station_assignment", (b, nc)),
        ("station_warehouse_assignment", (b, ns)),
    ):
        if key not in td or td[key].dtype != torch.int64 or tuple(td[key].shape) != shape:
            raise ValueError(f"{key} must be an int64 index tensor with shape {shape}")
    z, a = assignment_matrices(td)
    demand = td["customer_demand"].double()
    station_load = (z * demand[:, None, :]).sum(-1)
    warehouse_load = (a * station_load[:, None, :]).sum(-1)
    active_s, active_w = station_load > 0, warehouse_load > 0
    sw = td["station_warehouse_assignment"]
    violations = {
        "customer_single_source": (z.sum(1) != 1).any(-1),
        "active_station_single_source": ((a.sum(1) != 1) & active_s).any(-1),
        "inactive_station_assignment": ((sw != -1) & ~active_s).any(-1),
        "station_capacity": (station_load > td["station_capacity"].double() + atol).any(-1),
        "warehouse_capacity": (warehouse_load > td["warehouse_capacity"].double() + atol).any(-1),
    }
    for key, expected in (
        ("station_open", active_s),
        ("warehouse_open", active_w),
        ("station_load", station_load),
        ("warehouse_load", warehouse_load),
        ("station_remaining_capacity", td["station_capacity"].double() - station_load),
        ("warehouse_remaining_capacity", td["warehouse_capacity"].double() - warehouse_load),
    ):
        if key in td:
            if td[key].shape != expected.shape:
                raise ValueError(f"Invalid shape for {key}")
            if expected.dtype == torch.bool:
                if td[key].dtype != torch.bool:
                    raise ValueError(f"{key} must be bool")
                violations[key] = (td[key] != expected).any(-1)
            else:
                # float32 缓存允许一个舍入单位的误差；
                # 上方独立重算的容量约束仍使用绝对容差 atol。
                tolerance = atol + torch.finfo(torch.float32).eps * expected.abs()
                violations[key] = (
                    ~torch.isfinite(td[key]) | ((td[key].double() - expected).abs() > tolerance)
                ).any(-1)
    if "infeasible" in td:
        violations["infeasible"] = td["infeasible"].reshape(b).bool()
    valid = ~torch.stack(list(violations.values()), -1).any(-1)
    report = ValidityReport(valid, violations, station_load, warehouse_load)
    if raise_on_error and not report.all_valid:
        errors = {
            key: value.nonzero().flatten().tolist()
            for key, value in violations.items()
            if value.any()
        }
        raise ValueError(f"Invalid TSCFLP solution: {errors}")
    return report


def objective(td: TensorDict, solution=None, *, dtype=torch.float32, terms=None, numpy_state=None):
    # 根据不可拆分分配，以 float64 重新计算目标函数的四项成本。
    # 默认返回 float32，可通过 dtype 指定输出精度。未完成分配时返回
    # 已分配部分的成本，该数值本身不代表解可行。
    td = as_tensordict(td, solution)
    if terms is not None:
        # 混合整数线性规划的符号变量使用完全相同的四项成本。
        coefficients = {key: td[key][0].double().cpu().numpy() for key in terms}
        return sum(
            float(coefficients[key][index]) * variable
            for key, variables in terms.items()
            for index, variable in variables.items()
        )
    if numpy_state is not None:
        # 保持模拟退火的 float64 归约顺序，从而保持 Metropolis 接受轨迹。
        data, state = numpy_state
        active = np.flatnonzero(state.station_load > 0)
        return float(
            np.dot(data.d, data.csc[state.z, np.arange(data.nc)])
            + np.dot(state.station_load[active], data.cws[state.a[active], active])
            + data.g[state.station_load > 0].sum()
            + data.f[state.warehouse_load > 0].sum()
        )
    z, a = assignment_matrices(td)
    demand = td["customer_demand"].double()
    station_load = (z * demand[:, None, :]).sum(-1)
    warehouse_load = (a * station_load[:, None, :]).sum(-1)
    fixed_s = ((station_load > 0) * td["station_fixed_cost"].double()).sum(-1)
    fixed_w = ((warehouse_load > 0) * td["warehouse_fixed_cost"].double()).sum(-1)
    customer_cost = (z * demand[:, None, :] * td["cost_sc"].double()).sum((1, 2))
    warehouse_cost = (a * station_load[:, None, :] * td["cost_ws"].double()).sum((1, 2))
    return (fixed_s + fixed_w + customer_cost + warehouse_cost).to(dtype)


def compute_gap(cost: torch.Tensor, optimal_cost: torch.Tensor) -> torch.Tensor:
    # 计算相对于已证明最优值的间隙；缺失的最优值以 NaN 表示。
    ref = torch.as_tensor(optimal_cost, dtype=cost.dtype, device=cost.device)
    gap = 100 * (cost - ref) / ref.clamp_min(torch.finfo(cost.dtype).tiny)
    gap = torch.where((ref == 0) & (cost == 0), torch.zeros_like(gap), gap)
    gap = torch.where((ref == 0) & (cost > 0), torch.full_like(gap, float("inf")), gap)
    return torch.where(torch.isfinite(ref) & (ref >= 0), gap, torch.full_like(gap, float("nan")))


@dataclass(frozen=True)
class Instance:
    # 经校验的 float32 批量实例，不隐式转换数值或重新采样。

    tensors: TensorDict

    def __post_init__(self):
        validate_instance(self.tensors)

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return tuple(self.tensors[key].shape[-1] for key in STATIC_KEYS[:3])


@dataclass(frozen=True)
class Solution:
    # 使用整数索引表示不可拆分分配；-1 表示节点尚未分配。

    customer_station_assignment: torch.Tensor
    station_warehouse_assignment: torch.Tensor

    def attach(self, instance) -> TensorDict:
        td = instance.tensors if isinstance(instance, Instance) else instance
        state = td.select(*(key for key in STATIC_KEYS if key in td)).clone()
        state["customer_station_assignment"] = self.customer_station_assignment
        state["station_warehouse_assignment"] = self.station_warehouse_assignment
        return state


def as_tensordict(instance, solution=None) -> TensorDict:
    td = instance.tensors if isinstance(instance, Instance) else instance
    return solution.attach(td) if solution is not None else td


def path_increment_values(mass, cost_sc, cost_ws, fixed_s, fixed_w):
    # 保持原模拟退火和路径解码的运算顺序，避免改变舍入结果。
    delta = mass * (cost_sc + cost_ws)
    delta = delta + fixed_s
    return delta + fixed_w


def step_increment(td, rows, *, station, mass, customer, warehouse, new_s, new_w):
    # 路径动作的完整四项边际费用，保持原运算顺序与float64精度。
    return path_increment_values(
        mass,
        td["cost_sc"][rows, station, customer].double(),
        td["cost_ws"][rows, warehouse, station].double(),
        new_s * td["station_fixed_cost"][rows, station].double(),
        new_w * td["warehouse_fixed_cost"][rows, warehouse].double(),
    )


def sa_delta(
    data, state, customers, destinations, assignment, station_load, warehouse_load, touched
):
    # 集中保留原模拟退火边际成本的精确运算顺序，确保核算一致。
    delta_sc = 0.0
    if len(customers) == 1:
        k = customers[0]
        delta_sc = data.d[k] * (data.csc[destinations[0], k] - data.csc[state.z[k], k])
    elif len(customers) == 2:
        k, ell = customers
        first, second = state.z[k], state.z[ell]
        delta_sc = data.d[k] * (data.csc[second, k] - data.csc[first, k]) + data.d[ell] * (
            data.csc[first, ell] - data.csc[second, ell]
        )
    delta_ws, delta_fixed = 0.0, 0.0
    for j in touched:
        old, new = state.station_load[j], station_load[j]
        if old > 0:
            delta_ws -= old * data.cws[state.a[j], j]
        if new > 0:
            delta_ws += new * data.cws[assignment[j], j]
        delta_fixed += data.g[j] * (int(new > 0) - int(old > 0))
    delta_fixed += np.dot(
        data.f, (warehouse_load > 0).astype(np.int64) - (state.warehouse_load > 0).astype(np.int64)
    )
    return float(delta_sc + delta_ws + delta_fixed)

# 在满足容量约束的不可拆分分配方案上执行独立模拟退火。

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from ..problem import (
    STATIC_KEYS,
    check_feasible,
    objective,
    path_increment_values,
    sa_delta,
    validate_instance,
)


@dataclass(frozen=True)
class SAResult:
    feasible: bool
    cost: float | None
    initial_cost: float | None
    solve_time: float
    proposals: int
    feasible_proposals: int
    accepted_moves: int
    accepted_uphill_moves: int
    successful_starts: int
    solution: TensorDict | None


@dataclass
class _State:
    z: np.ndarray
    a: np.ndarray
    station_load: np.ndarray
    warehouse_load: np.ndarray
    cost: float


@dataclass
class _Move:
    customers: tuple[int, ...]
    stations: tuple[int, ...]
    assignment: np.ndarray
    station_load: np.ndarray
    warehouse_load: np.ndarray
    delta: float


class _Problem:
    def __init__(self, td: TensorDict) -> None:
        self.td = td
        self.d, self.qs, self.qw, self.g, self.f, self.csc, self.cws = (
            td[key][0].detach().double().cpu().numpy()
            for key in (
                "customer_demand",
                "station_capacity",
                "warehouse_capacity",
                "station_fixed_cost",
                "warehouse_fixed_cost",
                "cost_sc",
                "cost_ws",
            )
        )
        self.nc, self.ns, self.nw = len(self.d), len(self.qs), len(self.qw)

    def construct(self, rng: np.random.Generator, randomized: bool) -> _State | None:
        z, a = np.full(self.nc, -1, dtype=np.int64), np.full(self.ns, -1, dtype=np.int64)
        ds, lw = np.zeros(self.ns), np.zeros(self.nw)
        for k in np.argsort(-self.d, kind="stable"):
            mass = self.d[k]
            choices = []
            for j in range(self.ns):
                if ds[j] + mass > self.qs[j]:
                    continue
                warehouses = (int(a[j]),) if ds[j] > 0 else range(self.nw)
                for i in warehouses:
                    if lw[i] + mass <= self.qw[i]:
                        delta = path_increment_values(
                            mass,
                            self.csc[j, k],
                            self.cws[i, j],
                            self.g[j] if ds[j] == 0 else 0,
                            self.f[i] if lw[i] == 0 else 0,
                        )
                        choices.append((float(delta), j, i))
            if not choices:
                return None
            choices.sort()
            selected = int(rng.integers(min(3, len(choices)))) if randomized else 0
            _, j, i = choices[selected]
            z[k], a[j] = j, i
            ds[j] += mass
            lw[i] += mass
        state = _State(z, a, ds, lw, 0.0)
        state.cost = objective(self.td, numpy_state=(self, state))
        return state

    def propose(self, state: _State, rng: np.random.Generator) -> _Move | None:
        if self.ns == 1 and self.nw == 1:
            return None
        draw = float(rng.random())
        a = state.a.copy()
        ds = state.station_load.copy()
        customers: tuple[int, ...] = ()
        destinations: tuple[int, ...] = ()
        if self.nw > 1 and (self.ns == 1 or draw < 0.05):
            active = np.flatnonzero(ds > 0)
            if not len(active):
                return None
            j = int(rng.choice(active))
            i = int(rng.integers(self.nw - 1))
            i += i >= a[j]
            a[j] = i
            touched = (j,)
        elif self.ns > 1 and (draw < 0.60 or self.nc == 1):
            k = int(rng.integers(self.nc))
            source = int(state.z[k])
            target = int(rng.integers(self.ns - 1))
            target += target >= source
            mass = self.d[k]
            ds[source] -= mass
            ds[target] += mass
            if ds[target] > self.qs[target]:
                return None
            if ds[source] == 0:
                a[source] = -1
            if state.station_load[target] == 0:
                available = state.warehouse_load.copy()
                available[state.a[source]] -= mass
                warehouses = np.flatnonzero(available + mass <= self.qw)
                if not len(warehouses):
                    return None
                a[target] = int(rng.choice(warehouses))
            customers, destinations = (k,), (target,)
            touched = (source, target)
        else:
            if self.nc < 2:
                return None
            k = int(rng.integers(self.nc))
            ell = int(rng.integers(self.nc - 1))
            ell += ell >= k
            first, second = int(state.z[k]), int(state.z[ell])
            if first == second:
                return None
            mass_delta = self.d[ell] - self.d[k]
            ds[first] += mass_delta
            ds[second] -= mass_delta
            touched = (first, second)
            customers, destinations = (k, ell), (second, first)
        lw = state.warehouse_load.copy()
        for j in touched:
            old, new = state.station_load[j], ds[j]
            if new < 0 or new > self.qs[j]:
                return None
            if old > 0:
                lw[state.a[j]] -= old
            if new > 0:
                if a[j] < 0:
                    return None
                lw[a[j]] += new
        if (lw < 0).any() or (lw > self.qw).any():
            return None
        delta = sa_delta(self, state, customers, destinations, a, ds, lw, touched)
        return _Move(customers, destinations, a, ds, lw, delta)

    @staticmethod
    def apply(state: _State, move: _Move) -> None:
        for k, j in zip(move.customers, move.stations, strict=True):
            state.z[k] = j
        state.a = move.assignment
        state.station_load = move.station_load
        state.warehouse_load = move.warehouse_load
        state.cost += move.delta


def solve_simulated_annealing(
    td: TensorDict,
    seed: int = 123,
    iterations: int = 5000,
    restarts: int = 4,
    initial_temperature: float | None = None,
    final_temperature_ratio: float = 1e-3,
    initialization_attempts: int = 32,
) -> SAResult:
    # 对满足约束的不可拆分邻域移动使用几何降温和 Metropolis 接受准则。
    # 每次重启独立构造初始解，不使用神经网络或 Gurobi 的解。
    # 不可行的候选移动直接拒绝；初始化失败计入统计，可返回 feasible=False。
    # 不调用其他求解器，也不修复分配方案。
    started = time.perf_counter()
    validate_instance(td)
    if td.batch_size[0] != 1:
        raise ValueError("SA accepts one instance with batch shape [1]")
    if iterations < 1 or restarts < 1 or initialization_attempts < 1:
        raise ValueError("SA iteration/restart/initialization budgets must be positive")
    if not math.isfinite(final_temperature_ratio) or not 0 < final_temperature_ratio < 1:
        raise ValueError("final_temperature_ratio must lie strictly between 0 and 1")
    if initial_temperature is not None and (
        not math.isfinite(initial_temperature) or initial_temperature <= 0
    ):
        raise ValueError("initial_temperature must be finite and positive")
    problem = _Problem(td)
    rng = np.random.default_rng(seed)
    best = None
    initial_best = None
    proposals = feasible_proposals = accepted = uphill = starts = 0
    for restart in range(restarts):
        state = None
        for attempt in range(initialization_attempts):
            state = problem.construct(rng, randomized=restart > 0 or attempt > 0)
            if state is not None:
                break
        if state is None:
            continue
        starts += 1
        initial_best = state.cost if initial_best is None else min(initial_best, state.cost)
        if best is None or state.cost < best.cost:
            best = _State(
                state.z.copy(),
                state.a.copy(),
                state.station_load.copy(),
                state.warehouse_load.copy(),
                state.cost,
            )
        positive = []
        if initial_temperature is None:
            for _ in range(128):
                candidate = problem.propose(state, rng)
                if candidate is not None and candidate.delta > 0:
                    positive.append(candidate.delta)
        temperature = initial_temperature or (
            float(np.median(positive)) / -math.log(0.8) if positive else 1.0
        )
        for step in range(iterations):
            proposals += 1
            candidate = problem.propose(state, rng)
            if candidate is None:
                continue
            feasible_proposals += 1
            current_t = temperature * final_temperature_ratio ** (step / max(1, iterations - 1))
            if candidate.delta <= 0 or rng.random() < math.exp(-candidate.delta / current_t):
                problem.apply(state, candidate)
                accepted += 1
                uphill += candidate.delta > 0
                if accepted % 1000 == 0:
                    state.cost = objective(td, numpy_state=(problem, state))
                if state.cost < best.cost:
                    best = _State(
                        state.z.copy(),
                        state.a.copy(),
                        state.station_load.copy(),
                        state.warehouse_load.copy(),
                        state.cost,
                    )
    if best is None:
        return SAResult(
            False,
            None,
            None,
            time.perf_counter() - started,
            proposals,
            feasible_proposals,
            accepted,
            uphill,
            starts,
            None,
        )
    solution = td.select(*(key for key in STATIC_KEYS if key in td)).detach().cpu().clone()
    solution["customer_station_assignment"] = torch.from_numpy(best.z.copy())[None]
    solution["station_warehouse_assignment"] = torch.from_numpy(best.a.copy())[None]
    solution["station_load"] = torch.from_numpy(best.station_load.copy())[None].float()
    solution["warehouse_load"] = torch.from_numpy(best.warehouse_load.copy())[None].float()
    solution["station_open"] = solution["station_load"] > 0
    solution["warehouse_open"] = solution["warehouse_load"] > 0
    check_feasible(solution, raise_on_error=True)
    cost = float(objective(solution)[0])
    if abs(objective(td, numpy_state=(problem, best)) - best.cost) > 1e-5:
        raise RuntimeError("SA incremental objective drifted from independent recomputation")
    solution["incremental_cost"] = torch.tensor([cost], dtype=torch.float32)
    solution["done"] = torch.tensor([True])
    solution["infeasible"] = torch.tensor([False])
    solution["reward"] = -solution["incremental_cost"]
    return SAResult(
        True,
        cost,
        initial_best,
        time.perf_counter() - started,
        proposals,
        feasible_proposals,
        accepted,
        uphill,
        starts,
        solution,
    )


def main(argv=None):
    from .common import baseline_main

    baseline_main("sa", solve_simulated_annealing, argv)


if __name__ == "__main__":
    main()

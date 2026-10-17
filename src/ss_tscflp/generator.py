# 显式生成带有可行性构造保证的 NPZ 数据集。

from collections.abc import Sequence

import torch
from tensordict import TensorDict


class TSCFLPGenerator:
    # 生成批量形状为 [B] 的 float32 实例。
    # 先按需求降序将顾客不可拆分地分配到站点，再将站点负载同样分配到仓库。
    # 额外容量保留该可行分配，但不会向策略暴露这一构造结果。两级容量共用
    # U(1.3, 1.8) 松弛倍数，满足指定的边缘分布；单站点时也保持最大容量关系。

    def __init__(
        self,
        n_w: int = 5,
        n_s: int = 10,
        n_c: int = 20,
        cost_factor: float = 10.0,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if any(not isinstance(n, int) or n < 1 for n in (n_w, n_s, n_c)):
            raise ValueError("n_w, n_s and n_c must be positive integers")
        if not 0 <= cost_factor < float("inf"):
            raise ValueError("cost_factor must be finite and nonnegative")
        self.n_w, self.n_s, self.n_c = n_w, n_s, n_c
        self.cost_factor = float(cost_factor)
        self.device = torch.device(device)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(torch.initial_seed() if seed is None else seed)

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=self.rng, dtype=torch.float32)

    @staticmethod
    def _partition_loads(items: torch.Tensor, bins: int) -> torch.Tensor:
        # 将每个需求项整体分配到一个容器，构造满足容量约束的分配依据。
        loads = items.new_zeros(items.shape[0], bins)
        rows = torch.arange(items.shape[0])
        for item in items.sort(dim=-1, descending=True, stable=True).values.unbind(-1):
            selected = loads.argmin(dim=-1)
            loads[rows, selected] += item
        return loads

    def __call__(self, batch_size: int | Sequence[int] = (1,)) -> TensorDict:
        shape = [batch_size] if isinstance(batch_size, int) else list(batch_size)
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError("Only a nonempty leading batch shape [B] is supported")
        b = shape[0]
        demand = 1.0 + 9.0 * self._rand(b, self.n_c)
        total = demand.sum(-1, keepdim=True)
        slack = (0.3 + 0.5 * self._rand(b, 1)) * total
        station_base = self._partition_loads(demand, self.n_s)
        station_weight = 0.1 + self._rand(b, self.n_s)
        station_capacity = station_base + slack * station_weight / station_weight.sum(-1, True)
        warehouse_base = self._partition_loads(station_base, self.n_w)
        warehouse_weight = 0.1 + self._rand(b, self.n_w)
        warehouse_extra = slack * warehouse_weight / warehouse_weight.sum(-1, True)

        # 仅将未使用的松弛容量转移到基础负载最大的仓库。
        # 构造的可行负载不减少，总容量保持不变。
        rows = torch.arange(b)
        target = warehouse_base.argmax(-1)
        target_capacity = (warehouse_base + warehouse_extra)[rows, target]
        missing = (station_capacity.max(-1).values - target_capacity).clamp_min(0)
        donors = warehouse_extra.clone()
        donors[rows, target] = 0
        donor_sum = donors.sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
        warehouse_extra -= donors * (missing / donor_sum).unsqueeze(-1)
        warehouse_extra[rows, target] += missing
        warehouse_capacity = warehouse_base + warehouse_extra

        w_loc = self._rand(b, self.n_w, 2)
        s_loc = self._rand(b, self.n_s, 2)
        c_loc = self._rand(b, self.n_c, 2)
        td = TensorDict(
            {
                "warehouse_capacity": warehouse_capacity,
                "station_capacity": station_capacity,
                "customer_demand": demand,
                "warehouse_fixed_cost": 50.0 + 50.0 * self._rand(b, self.n_w),
                "station_fixed_cost": 20.0 + 30.0 * self._rand(b, self.n_s),
                "cost_ws": torch.cdist(w_loc, s_loc) * self.cost_factor,
                "cost_sc": torch.cdist(s_loc, c_loc) * self.cost_factor,
                "warehouse_loc": w_loc,
                "station_loc": s_loc,
                "customer_loc": c_loc,
            },
            batch_size=[b],
        )
        return td.to(self.device)


def main(argv=None):
    import argparse
    from pathlib import Path

    from .common import parse_config, set_seed
    from .dataset import save_instances

    parser = argparse.ArgumentParser(description="显式生成带有可行性构造保证的 NPZ 数据集。")
    for name in ("n_w", "n_s", "n_c", "train_size", "val_size", "test_size", "seed", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--cost-factor", type=float)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parse_config(parser, argv)
    required = (
        "device",
        "seed",
        "output",
        "n_w",
        "n_s",
        "n_c",
        "train_size",
        "val_size",
        "test_size",
    )
    if any(key not in args for key in required):
        parser.error(f"Required settings: {', '.join(required)}")
    if args["device"] != "cpu":
        raise ValueError("Dataset generation uses the original CPU generator; specify device=cpu")
    if min(args[key] for key in ("train_size", "val_size", "test_size")) < 1:
        raise ValueError("Split sizes must be positive")
    torch.set_num_threads(args.get("threads", 1))
    set_seed(args["seed"], device=args["device"])
    root = Path(args["output"])
    paths = {split: root / split / "instances.npz" for split in ("train", "val", "test")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    settings = {key: args[key] for key in ("n_w", "n_s", "n_c")}
    settings["cost_factor"] = args.get("cost_factor", 10.0)
    for offset, (split, path) in enumerate(paths.items()):
        seed = args["seed"] + offset
        data = TSCFLPGenerator(**settings, seed=seed, device="cpu")([args[f"{split}_size"]])
        save_instances(data, path, metadata={"split": split, "seed": seed, **settings})
        print(f"{split}: {data.batch_size[0]} instances -> {path}")


if __name__ == "__main__":
    main()

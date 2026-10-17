# 数据集格式

使用 `python scripts/run_generate.py` 显式生成数据。默认创建 `train/instances.npz`、`val/instances.npz` 和 `test/instances.npz`，各划分使用不同的随机种子。生成器不会覆盖已有文件。

实例数值字段均为有限、非负的 **float32** 数组，首维为批量大小 B。顾客需求必须严格为正。加载数据时不归一化、不重新缩放，也不重新采样。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `warehouse_capacity` | B × W | 仓库容量 |
| `station_capacity` | B × S | 站点容量 |
| `customer_demand` | B × C | 不可拆分的顾客需求 |
| `warehouse_fixed_cost` | B × W | 仓库启用固定成本 |
| `station_fixed_cost` | B × S | 站点启用固定成本 |
| `cost_ws` | B × W × S | 仓库至站点的单位需求运输成本 |
| `cost_sc` | B × S × C | 站点至顾客的单位需求运输成本 |
| `warehouse_loc` | B × W × 2 | 可选的仓库坐标 |
| `station_loc` | B × S × 2 | 可选的站点坐标 |
| `customer_loc` | B × C × 2 | 可选的顾客坐标 |
| `metadata_json` | Unicode 标量 | 可选 JSON 对象，记录划分、种子和来源 |

`NPZDataset` 校验上述数据结构。`make_dataloader` 使用显式随机种子，提供可复现的打乱批次。固定数据集训练会检查各划分的维度、数量、种子重叠及重复数值行；报告中的 SHA-256 哈希用于标识实际输入文件。

```python
from ss_tscflp.dataset import NPZDataset, make_dataloader

dataset = NPZDataset("data/train/instances.npz")
loader = make_dataloader(dataset, batch_size=8, shuffle=True, seed=42)
for batch in loader:
    print(batch.batch_size)
```

生成器保持原有的可行实例构造分布、容量松弛分布及成本尺度。即使实例存在可行解，某次解码或构造式启发算法仍可能进入死路，失败结果会明确保留。

分配方案使用 int64 的 `customer_station_assignment`（B × C）和 `station_warehouse_assignment`（B × S），其中 `-1` 表示尚未分配。方案存档还包含 `feasible`、`cost`，以及记录源数据集哈希的 `metadata_json`。

不可行行的成本仅表示已分配部分，不得计入可行解平均成本。检查器会根据实际分配独立重算可行性和成本，不信任存档中的汇总字段。

模型检查点同样采用 NPZ，使用 Unicode 标量 `manifest_json` 和数值张量数组；其结构与数据集、分配方案不同。所有加载器均使用 `allow_pickle=False`，不接受对象数组或历史 pickle 检查点。

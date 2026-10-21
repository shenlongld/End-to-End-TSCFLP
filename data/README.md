# 
模型的最佳权重为145epoch.npz

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


```python
from ss_tscflp.dataset import NPZDataset, make_dataloader

dataset = NPZDataset("data/train/instances.npz")
loader = make_dataloader(dataset, batch_size=8, shuffle=True, seed=42)
for batch in loader:
    print(batch.batch_size)
```


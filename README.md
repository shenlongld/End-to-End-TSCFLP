# SS-TSCFLP

面向**单源两级容量约束设施选址问题**（Single-Source Two-Echelon Capacitated Facility Location Problem）的可复现求解工具，包含两阶段神经策略、耦合路径神经策略、模拟退火（SA），以及可选的 Gurobi 精确求解基线。

## 安装

支持 Python **3.11–3.13**，推荐使用 **3.12**。先创建并激活独立虚拟环境，再从仓库根目录安装：

```text
python -m venv .venv
```

| 平台 | 激活命令 |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows 命令提示符 | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

```text
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

只需部署运行时，可使用 `python -m pip install .`。项目采用 `src/` 包结构，安装后无需设置 `PYTHONPATH`，也不依赖原始实验目录。主要依赖版本固定，与原科研运行环境保持一致。

使用 GPU 时，先安装与所固定 PyTorch 版本兼容的 CUDA 构建，再安装本项目，并显式设置 `device: cuda:0`。指定设备不可用时直接报错，不自动切换设备。

## 快速开始

所有运行脚本均为 Python 文件。安装后，在仓库根目录执行：

```text
python scripts/run_generate.py
python scripts/run_train.py --preset smoke
python scripts/run_infer.py --checkpoint checkpoints/smoke/best.npz --data data/test/instances.npz --device cpu --decode-type greedy --output results/smoke
```

`smoke` 预设使用小模型和小规模训练实例，适合验证完整流程。策略支持不同实例规模。运行标准的 **3 个仓库、10 个站点、100 个顾客**实验：

```text
python scripts/run_train.py
python scripts/run_eval.py
```

评估会导出 `report.json`、逐实例的 `results.csv`，以及每种方法对应的分配方案 `.npz`。失败实例单独统计，不计入可行解平均成本和最优间隙。采样推理对每个实例恰好生成 K 个候选，最后一个分块不足整块时仍按剩余数量执行。

安装后还可使用命令行入口：`ss-tscflp-generate`、`ss-tscflp-train`、`ss-tscflp-infer`、`ss-tscflp-evaluate`、`ss-tscflp-sa`、`ss-tscflp-gurobi` 和 `ss-tscflp-benchmark`。也支持 `python -m ss_tscflp.train` 等模块形式，通过 `--help` 查看参数。

## 配置与路径

- YAML 中的路径相对**该配置文件所在目录**解析；命令行路径相对**当前工作目录**解析。支持绝对路径及带空格、中文的路径，在命令行中应为含空格的路径加引号。
- 预设脚本相对自身位置查找配置文件，因此可以从其他工作目录启动。安装后的命令行入口通过 `--config path/to/train.yaml` 指定配置文件。
- 文件缺失、未知配置项、不支持的设备或不兼容的检查点都会直接报错，不因路径缺失而自动生成数据。
- `data_mode: synthetic` 显式启用原有的逐轮新实例训练流。`data_mode: npz` 必须同时提供 `dataset_dir`，使用完整的固定训练集和验证集；实例数量与问题维度必须匹配训练配置。使用 NPZ 模式前需显式生成数据。
- 数据、检查点、日志和结果默认不纳入 Git。需要共享的产物应单独上传，并显式传入路径；Gurobi 许可证不应提交到仓库。

标准训练配置见 [configs/train.yaml](configs/train.yaml)，数据生成和评估配置分别见 [configs/generator.yaml](configs/generator.yaml) 与 [configs/eval.yaml](configs/eval.yaml)。

## 训练与复现

**path** 策略保留原始分组 REINFORCE 的留一法基线、均值与 Max@K 混合目标、对样本求和的 Max@K 估计量、熵正则、预热阶段、零权重衰减的 AdamW 及梯度裁剪。

**two_stage** 策略保留原生 RL4CO REINFORCE、贪心 rollout 基线和 Adam 优化器。两套网络结构及其消融配置均保留在 `model.py` 中。

```text
ss-tscflp-train --method two_stage --device cpu --data-mode synthetic --epochs 2 --train-size 16 --val-size 16 --batch-size 8 --output checkpoints/two_stage
ss-tscflp-train --method path --resume checkpoints/path/last.npz --epochs 60 --device cpu
```

统一通过 `ss_tscflp.common.set_seed` 设置 Python、NumPy 和 PyTorch 的随机种子，并启用确定性运算。检查点以 NPZ 保存数值张量和 JSON 元数据，包含优化器、私有生成器、验证数据、两阶段基线及 PyTorch 随机状态，不使用 pickle。

`last.npz` 支持在训练轮次边界精确续训，`best.npz` 用于保存最优策略的推理快照。续训允许调整输出目录、设备和总轮数，也允许将内容相同的 NPZ 数据迁移到其他路径；数据哈希和训练配置会被校验。逐位一致性已在相同 CPU 运行环境中验证，不同硬件或依赖版本可能产生浮点差异。

本版本不加载历史 `.pt` 文件。历史实验可单独保留，新训练统一生成可迁移的 `.npz` 检查点。

## 数学定义与求解约定

设 i、j、k 分别表示仓库、站点和顾客。顾客需求 d[k] 通过 z[j,k] 整体分配给一个站点，每个启用站点通过 a[i,j] 整体分配给一个仓库。D[j] 表示站点 j 承担的总需求，最小化以下目标：

```text
sum_i f[i] x[i] + sum_j g[j] y[j]
+ sum_jk d[k] c_sc[j,k] z[j,k]
+ sum_ij D[j] c_ws[i,j] a[i,j]
```

两级设施容量均为硬约束，设施启用当且仅当其负载为正；未启用站点的仓库索引为 `-1`。顾客需求严格为正，顾客需求和站点出库负载均不可拆分。神经解码进入死路时保留原有固定惩罚，不修复分配方案。

`problem.py` 统一定义 `Instance`、`Solution`、`objective`、`check_feasible` 和 `compute_gap`。神经策略奖励、SA 输出及 Gurobi 解提取共用这些函数，增量成本核算也集中在该文件中。动作掩码和邻域构造保持原有行为。

Gurobi 的线性化模型表达相同的不可拆分约束，符号目标由 `problem.objective` 构造。解码器中的货币特征仅作为策略输入，不额外修改奖励或数学问题。

最优间隙只使用**已证明的最优值**作为基准：`100 * (cost - optimum) / optimum`。限时内的当前可行解单独标记，缺失最优基准时不报告间隙；最优值和当前成本均为零时为 0%，最优值为零但当前成本为正时为无穷大。

## 基线与性能测试

```text
ss-tscflp-sa --data data/test/instances.npz --device cpu --iterations 5000 --restarts 4 --output results/sa
python -m pip install ".[gurobi]"
ss-tscflp-gurobi --data data/test/instances.npz --device cpu --time-limit 60 --output results/gurobi
ss-tscflp-gurobi --data data/test/instances.npz --solution results/gurobi/solutions.npz --device cpu
python scripts/run_eval.py --methods greedy sampling sa gurobi
python scripts/run_benchmark.py --data data/test/instances.npz --checkpoint checkpoints/path/best.npz --device cpu --batch-size 1 --warmup 1 --repeats 5 --output results/latency.json
```

Gurobi 求解需要安装 `gurobipy` 并配置有效许可证，分配方案检查器本身不依赖 Gurobi。SA 和 Gurobi 使用 CPU。

性能测试在选用 CUDA 时显式同步设备，计时不包含文件读取和主机与设备之间的数据传输。报告分别给出批次延迟的 p50/p95、吞吐量和摊销到单个实例的时间；测量单请求延迟时使用 `--batch-size 1`。

## 验证

```text
python -m pytest
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m build
```

测试覆盖独立目标值与可行性校验、微型实例穷举与 Gurobi 对照、两种解码过程、网络梯度与等变性、分组损失的精确梯度、NPZ 读写往返、可复现 SA 及训练轮次边界续训。Gurobi 不可用时，仅相关求解测试跳过。数据交换格式见 [data/README.md](data/README.md)。源码说明统一使用中文 `#` 注释。

## 首次推送到 GitHub

在尚未配置 `origin` 的本地仓库根目录执行，目标为 [shenlongld/End-to-End-TSCFLP](https://github.com/shenlongld/End-to-End-TSCFLP)：

```text
git add .
git commit -m "发布 SS-TSCFLP 中文版"
git branch -M main
git remote add origin https://github.com/shenlongld/End-to-End-TSCFLP.git
git push -u origin main
```

如果本地已经配置 `origin`，将添加远端的命令改为 `git remote set-url origin https://github.com/shenlongld/End-to-End-TSCFLP.git`。后续更新通常只需执行 `git add .`、`git commit` 和 `git push`。

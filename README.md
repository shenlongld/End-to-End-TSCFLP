# SS-TSCFLP

面向**单源两级容量约束设施选址问题**（SS-TSCFLP）的端到端求解工具。

---

## 目录
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [脚本与运行指令](#脚本与运行指令)
  - [1. 数据生成](#1-数据生成)
  - [2. 模型训练与续训](#2-模型训练与续训)
  - [3. 模型推理与综合评估](#3-模型推理与综合评估)
  - [4. 基准算法（SA 与 Gurobi）](#4-基准算法sa-与-gurobi)
  - [5. 性能测速（Latency Benchmark）](#5-性能测速latency-benchmark)
- [配置说明](#配置说明)

---

## 环境安装

支持 Python **3.11–3.13**（推荐 **3.12**）。在独立虚拟环境中执行：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell
# .venv\Scripts\activate.bat    # Windows CMD

# 安装基础依赖
python -m pip install --upgrade pip
python -m pip install -e .

# （可选）如需使用 Gurobi 求解基线
python -m pip install ".[gurobi]"
```

---

## 快速开始

按顺序执行下列指令，完成一次可行性测试：

```bash
# 1. 生成示例数据
python scripts/run_generate.py

# 2. 快速训练
python scripts/run_train.py --preset smoke

# 3. 执行推理
python scripts/run_infer.py \
  --checkpoint checkpoints/smoke/best.npz \
  --data data/test/instances.npz \
  --device cpu \
  --decode-type greedy \
  --output results/smoke
```

---

## 脚本与运行指令

项目支持**脚本调用**（`python scripts/run_*.py`）与安装后的 **CLI 命令行入口**（`ss-tscflp-*`）。

### 1. 数据生成

```bash
# 使用默认配置生成数据
python scripts/run_generate.py

# CLI 方式：指定配置文件生成
ss-tscflp-generate --config configs/generator.yaml
```

---

### 2. 模型训练与续训

支持 **两阶段策略 (`two_stage`)** 与 **耦合路径策略 (`path`)**。

```bash
# 运行标准规模训练（3 仓库, 10 站点, 100 顾客）
python scripts/run_train.py

# 训练策略（CPU，在线合成数据模式）
ss-tscflp-train \
  --method two_stage \
  --device cpu \
  --data-mode synthetic \
  --epochs 2 \
  --train-size 16 \
  --val-size 16 \
  --batch-size 8 \
  --output checkpoints/two_stage

# 续训 path 策略（指定检查点并更新轮次）
ss-tscflp-train \
  --method path \
  --resume checkpoints/path/last.npz \
  --epochs 60 \
  --device cpu
```

---

### 3. 模型推理与综合评估

#### 单独模型推理（Infer）
```bash
python scripts/run_infer.py \
  --checkpoint checkpoints/path/best.npz \
  --data data/test/instances.npz \
  --device cpu \
  --decode-type sampling \
  --output results/infer_sampling
```

#### 跨方法横向评估（Evaluate）
对多种方法进行对比测试，自动导出 `report.json`、逐实例明细 `results.csv` 及方案文件 `.npz`：

```bash
# 联合评估 greedy、sampling、sa 与 gurobi
python scripts/run_eval.py --methods greedy sampling sa gurobi

# 或使用 CLI 指定配置文件进行评估
ss-tscflp-evaluate --config configs/eval.yaml
```

---

### 4. 基准算法（SA 与 Gurobi）

#### 模拟退火（Simulated Annealing）
```bash
ss-tscflp-sa \
  --data data/test/instances.npz \
  --device cpu \
  --iterations 5000 \
  --restarts 4 \
  --output results/sa
```

#### Gurobi 精确求解与解校验
*需安装 `gurobipy` 并配置有效许可证。*

```bash
# 求解并限制运行时间（例如 60 秒）
ss-tscflp-gurobi \
  --data data/test/instances.npz \
  --device cpu \
  --time-limit 60 \
  --output results/gurobi

# 校验已有解的合法性及目标值（不调用优化器，仅验证方案）
ss-tscflp-gurobi \
  --data data/test/instances.npz \
  --solution results/gurobi/solutions.npz \
  --device cpu
```

---

### 5. 性能测速（Latency Benchmark）

测量单请求延迟（`--batch-size 1`）或批量吞吐量。使用 CUDA 时会自动执行设备同步以确保计时准确：

```bash
python scripts/run_benchmark.py \
  --data data/test/instances.npz \
  --checkpoint checkpoints/path/best.npz \
  --device cpu \
  --batch-size 1 \
  --warmup 1 \
  --repeats 5 \
  --output results/latency.json
```

---

## 配置说明

配置文件集中在 `configs/` 目录下：

| 配置文件 | 说明 |
| --- | --- |
| `configs/train.yaml` | 模型架构、学习率、批大小、优化器及轮次配置 |
| `configs/generator.yaml` | 仓库、站点和顾客规模，容量及需求分布配置 |
| `configs/eval.yaml` | 评测数据集、比对算法清单、解码参数及超时设定 |

- **路径解析规则**：YAML 中的相对路径相对于**该配置文件所在目录**解析；命令行中传入的路径均相对于**当前终端工作目录**解析。
- **数据加载模式**：
  - `data_mode: synthetic`：每一轮动态在线合成随机实例训练。
  - `data_mode: npz`：使用静态数据集训练，须配合 `dataset_dir` 提供完整的训练/验证 NPZ 文件。
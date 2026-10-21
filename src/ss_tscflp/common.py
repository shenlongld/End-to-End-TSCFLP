# 显式运行配置，以及统一设置全局随机种子的唯一入口。

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def validate_device(device: str | torch.device) -> torch.device:
    if not str(device):
        raise ValueError("device must be explicitly set to cpu or cuda[:index]")
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda[:index]")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {index}")
    return resolved


def set_seed(seed: int, *, device: str | torch.device) -> None:
    # 统一设置 Python、NumPy 和 PyTorch 的种子，并启用确定性运算。
    validate_device(device)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def parse_config(parser: argparse.ArgumentParser, argv=None) -> dict:
    # 使用 YAML 默认配置，并允许命令行显式覆盖；未知配置项直接报错。
    parser.add_argument("--config", type=Path)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    settings = {}
    if path is not None:
        with path.open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)
        if not isinstance(settings, dict):
            raise ValueError("Config must be a YAML mapping")
        unknown = set(settings) - set(args)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        # 配置内的路径相对配置文件解析，切换工作目录后仍可使用。
        for key in (
            "data",
            "dataset_dir",
            "checkpoint",
            "output",
            "resume",
            "solution",
        ):
            if settings.get(key) is not None:
                value = Path(settings[key]).expanduser()
                settings[key] = str((path.resolve().parent / value).resolve())
    settings.update({key: value for key, value in args.items() if value is not None})
    return settings

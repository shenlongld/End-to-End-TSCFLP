# 按预设生成训练、验证和测试数据，支持命令行覆盖参数。

import sys
from pathlib import Path

from ss_tscflp.generator import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = [
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/generator.yaml"),
            *args,
        ]
    main(args)

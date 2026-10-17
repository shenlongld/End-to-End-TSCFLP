# 运行预设评估流程，导出逐实例的 CSV 和 JSON 结果。

import sys
from pathlib import Path

from ss_tscflp.evaluate import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = ["--config", str(Path(__file__).resolve().parents[1] / "configs/eval.yaml"), *args]
    main(args)

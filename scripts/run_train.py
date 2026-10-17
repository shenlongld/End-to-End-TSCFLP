# 使用标准配置或小规模冒烟测试配置运行训练。

import argparse
from pathlib import Path

from ss_tscflp.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=("standard", "smoke"), default="standard")
    preset, rest = parser.parse_known_args()
    if not any(arg == "--config" or arg.startswith("--config=") for arg in rest):
        rest = ["--config", str(Path(__file__).resolve().parents[1] / "configs/train.yaml"), *rest]
    if preset.preset == "smoke":
        # 用户参数放在最后，可覆盖预设值。
        rest = [
            "--epochs",
            "2",
            "--train-instances",
            "8",
            "--batch-instances",
            "2",
            "--group-size",
            "3",
            "--val-size",
            "4",
            "--val-samples",
            "3",
            "--sample-chunk",
            "2",
            "--n-w",
            "2",
            "--n-s",
            "3",
            "--n-c",
            "5",
            "--embed-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--output",
            str(Path(__file__).resolve().parents[1] / "checkpoints/smoke"),
            *rest,
        ]
    main(rest)

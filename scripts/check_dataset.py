from __future__ import annotations

import argparse
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import collect_dataset_summary, format_summary_lines, resolve_dataset_root
else:
    from ..gap_core import collect_dataset_summary, format_summary_lines, resolve_dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 YOLO 数据集目录与标签格式。")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/gap_dataset/data.yaml"),
        help="数据集根目录或 data.yaml 路径，默认 data/gap_dataset/data.yaml",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        dataset_root = resolve_dataset_root(args.data)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        print("请先准备数据集目录，或确认 --data 参数是否正确。")
        return 1

    print(f"正在检查数据集: {dataset_root}")
    summary = collect_dataset_summary(dataset_root)
    for line in format_summary_lines(summary):
        print(line)
    return 0 if summary.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

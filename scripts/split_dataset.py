from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import IMAGE_SUFFIXES, ensure_directories, gather_image_label_pairs
else:
    from ..gap_core import IMAGE_SUFFIXES, ensure_directories, gather_image_label_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 YOLO Detection 格式划分训练集和验证集。")
    parser.add_argument("--images", type=Path, required=True, help="原始图片目录")
    parser.add_argument("--labels", type=Path, required=True, help="原始标签目录")
    parser.add_argument("--out", type=Path, default=Path("data/gap_dataset"), help="输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    return parser


def copy_pair(image_path: Path, label_path: Path, out_root: Path, split: str) -> None:
    image_out = out_root / "images" / split / image_path.name
    label_out = out_root / "labels" / split / label_path.name
    shutil.copy2(image_path, image_out)
    shutil.copy2(label_path, label_out)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.images.exists():
        print(f"[ERROR] 图片目录不存在: {args.images}")
        return 1
    if not args.labels.exists():
        print(f"[ERROR] 标签目录不存在: {args.labels}")
        return 1
    if not 0 < args.val_ratio < 1:
        print("[ERROR] --val-ratio 必须在 0 和 1 之间。")
        return 1

    pairs, warnings = gather_image_label_pairs(args.images, args.labels)
    if warnings:
        for warning in warnings:
            print(f"[WARNING] {warning}")
    if not pairs:
        print("[ERROR] 没有找到可用于划分的数据对。请确认图片与标签是否同名。")
        return 1

    ensure_directories(
        [
            args.out / "images" / "train",
            args.out / "images" / "val",
            args.out / "labels" / "train",
            args.out / "labels" / "val",
        ]
    )

    random.seed(args.seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    if not train_items and val_items:
        train_items.append(val_items.pop())

    for pair in train_items:
        copy_pair(pair.image_path, pair.label_path, args.out, "train")
    for pair in val_items:
        copy_pair(pair.image_path, pair.label_path, args.out, "val")

    train_total = len(train_items)
    val_total = len(val_items)
    print(f"原始图片目录: {args.images}")
    print(f"原始标签目录: {args.labels}")
    print(f"输出目录: {args.out}")
    print(f"支持图片后缀: {', '.join(sorted(IMAGE_SUFFIXES))}")
    print(f"训练集数量: {train_total}")
    print(f"验证集数量: {val_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

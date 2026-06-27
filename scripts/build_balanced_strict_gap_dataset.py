from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from balanced_strict_gap_dataset import build_balanced_training_plan, make_augmented_stem
    from gap_core import ensure_directories
    from strict_gap_dataset import collect_strict_items, parse_label_rows_strict, relabel_rows_strict
else:
    from ..balanced_strict_gap_dataset import build_balanced_training_plan, make_augmented_stem
    from ..gap_core import ensure_directories
    from ..strict_gap_dataset import collect_strict_items, parse_label_rows_strict, relabel_rows_strict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构造类别均衡版严格 gap 数据集。")
    parser.add_argument("--source", type=Path, default=Path("../dataset"), help="原始数据集根目录")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/gap_dataset_strict_balanced"),
        help="输出目录",
    )
    parser.add_argument(
        "--target-train-per-class",
        type=int,
        default=24,
        help="训练集每类目标图片数；少数类通过安全增强补齐。",
    )
    return parser


def _augment_image(image, variant: str):
    if variant == "orig":
        return image
    result = image.copy()
    if "flip" in variant:
        result = cv2.flip(result, 1)
    if "bright" in variant:
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=20)
    return result


def _augment_rows(rows: list[list[str]], variant: str, class_id: int) -> list[str]:
    augmented = []
    for row in rows:
        _, x_center, y_center, width, height = row
        x = float(x_center)
        if "flip" in variant:
            x = 1.0 - x
        augmented.append(f"{class_id} {x:.6f} {float(y_center):.6f} {float(width):.6f} {float(height):.6f}")
    return augmented


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd()
    source_root = args.source if args.source.is_absolute() else (project_root / args.source).resolve()
    out_root = args.out if args.out.is_absolute() else (project_root / args.out).resolve()
    if not source_root.exists():
        print(f"[ERROR] 原始数据集目录不存在: {source_root}")
        return 1

    items = collect_strict_items(source_root, per_class_limits={0: 999999, 1: 999999, 2: 999999})
    train_items = [item for item in items if item.split == "train"]
    val_items = [item for item in items if item.split == "val"]

    training_plan = build_balanced_training_plan(train_items, target_per_class=args.target_train_per_class)
    if any(len(class_plan) < args.target_train_per_class for class_plan in training_plan.values()):
        print("[ERROR] 某些类别没有可用严格训练样本，无法构造均衡训练集。")
        return 1

    ensure_directories(
        [
            out_root / "images" / "train",
            out_root / "images" / "val",
            out_root / "images" / "test",
            out_root / "labels" / "train",
            out_root / "labels" / "val",
        ]
    )

    (out_root / "data.yaml").write_text(
        "\n".join(
            [
                "path: data/gap_dataset_strict_balanced",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: single_gap",
                "  1: multi_gap",
                "  2: truncated",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for subdir in [
        out_root / "images" / "train",
        out_root / "images" / "val",
        out_root / "images" / "test",
        out_root / "labels" / "train",
        out_root / "labels" / "val",
    ]:
        for path in subdir.iterdir():
            if path.is_file():
                path.unlink()

    split_counts = Counter()
    class_counts = Counter()

    for class_id, plan_entries in training_plan.items():
        for entry in plan_entries:
            item = entry["source"]
            variant = entry["variant"]
            rows = parse_label_rows_strict(item.label_path)
            image = cv2.imread(str(item.image_path))
            image_aug = _augment_image(image, variant)
            stem = item.stem if variant == "orig" else make_augmented_stem(item.stem, variant)
            image_ext = item.image_path.suffix
            image_target = out_root / "images" / "train" / f"{stem}{image_ext}"
            label_target = out_root / "labels" / "train" / f"{stem}.txt"
            cv2.imwrite(str(image_target), image_aug)
            label_target.write_text("\n".join(_augment_rows(rows, variant, class_id)) + "\n", encoding="utf-8")
            split_counts["train"] += 1
            class_counts[class_id] += len(rows)

    for item in val_items:
        shutil.copy2(item.image_path, out_root / "images" / "val" / item.image_path.name)
        rows = parse_label_rows_strict(item.label_path)
        relabeled = relabel_rows_strict(rows, item.class_id)
        (out_root / "labels" / "val" / f"{item.stem}.txt").write_text("\n".join(relabeled) + "\n", encoding="utf-8")
        split_counts["val"] += 1
        class_counts[item.class_id] += len(rows)

        shutil.copy2(item.image_path, out_root / "images" / "test" / item.image_path.name)
        split_counts["test"] += 1

    print(f"已生成类别均衡版严格数据集: {out_root}")
    print(f"样本数: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    print(
        "训练/验证标签框数量: "
        f"single_gap={class_counts[0]}, multi_gap={class_counts[1]}, truncated={class_counts[2]}"
    )
    print("策略: 保留严格 val/test，训练集对少数类使用水平翻转与轻度亮度增强补齐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

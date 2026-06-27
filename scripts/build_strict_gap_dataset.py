from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import ensure_directories
    from strict_gap_dataset import collect_strict_items, parse_label_rows_strict, plan_strict_splits, relabel_rows_strict
else:
    from ..gap_core import ensure_directories
    from ..strict_gap_dataset import collect_strict_items, parse_label_rows_strict, plan_strict_splits, relabel_rows_strict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构造更严格清洗后的高精度 gap 子集。")
    parser.add_argument("--source", type=Path, default=Path("../dataset"), help="原始数据集根目录")
    parser.add_argument("--out", type=Path, default=Path("data/gap_dataset_strict"), help="输出目录")
    parser.add_argument("--train-per-class", type=int, default=8, help="每类训练样本数")
    parser.add_argument("--val-per-class", type=int, default=3, help="每类验证样本数")
    parser.add_argument("--test-per-class", type=int, default=2, help="每类测试样本数")
    parser.add_argument(
        "--use-all-available",
        action="store_true",
        help="使用全部符合严格规则的原始 train/val 样本；test 将镜像严格 val 图片作为预测输入示例。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd()
    source_root = args.source if args.source.is_absolute() else (project_root / args.source).resolve()
    out_root = args.out if args.out.is_absolute() else (project_root / args.out).resolve()
    if not source_root.exists():
        print(f"[ERROR] 原始数据集目录不存在: {source_root}")
        return 1

    per_class_limits = (
        {0: 999999, 1: 999999, 2: 999999}
        if args.use_all_available
        else {
            0: args.train_per_class + args.val_per_class + args.test_per_class,
            1: args.train_per_class + args.val_per_class + args.test_per_class,
            2: args.train_per_class + args.val_per_class + args.test_per_class,
        }
    )
    items = collect_strict_items(source_root, per_class_limits=per_class_limits)

    by_class: dict[int, list] = defaultdict(list)
    for item in items:
        by_class[item.class_id].append(item)

    if not args.use_all_available:
        for class_id, required_total in per_class_limits.items():
            actual = len(by_class[class_id])
            if actual < required_total:
                print(f"[ERROR] 类别 {class_id} 样本不足，需要 {required_total}，实际 {actual}")
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

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: data/gap_dataset_strict",
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
    split_plan = plan_strict_splits(
        items,
        use_all_available=args.use_all_available,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
    )

    for split_name, selected_items in split_plan.items():
        for item in selected_items:
            image_target = out_root / "images" / split_name / item.image_path.name
            shutil.copy2(item.image_path, image_target)
            if split_name != "test":
                rows = parse_label_rows_strict(item.label_path)
                relabeled = relabel_rows_strict(rows, item.class_id)
                label_target = out_root / "labels" / split_name / f"{item.stem}.txt"
                label_target.write_text("\n".join(relabeled) + "\n", encoding="utf-8")
                class_counts[item.class_id] += len(relabeled)
            split_counts[split_name] += 1

    print(f"已生成严格清洗子集: {out_root}")
    print(f"样本数: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    print(
        "标签框数量: "
        f"single_gap={class_counts[0]}, multi_gap={class_counts[1]}, truncated={class_counts[2]}"
    )
    print("严格规则: 单目标居中且适中框=single_gap; 2-3 个居中适中框=multi_gap; 单目标贴边且面积足够大=truncated。")
    if args.use_all_available:
        print("当前模式: 使用全部严格合格样本，保留原始 train/val，test 镜像严格 val 图片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

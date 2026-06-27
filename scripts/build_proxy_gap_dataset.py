from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import ensure_directories
    from proxy_gap_dataset import collect_proxy_items, parse_label_rows, relabel_rows
else:
    from ..gap_core import ensure_directories
    from ..proxy_gap_dataset import collect_proxy_items, parse_label_rows, relabel_rows


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从现有检测数据中构造三类代理 gap 数据集。")
    parser.add_argument("--source", type=Path, default=Path("../dataset"), help="原始数据集根目录")
    parser.add_argument("--out", type=Path, default=Path("data/gap_dataset"), help="输出数据集目录")
    parser.add_argument("--train-per-class", type=int, default=60, help="每类训练样本数")
    parser.add_argument("--val-per-class", type=int, default=15, help="每类验证样本数")
    parser.add_argument("--test-per-class", type=int, default=10, help="每类测试样本数")
    parser.add_argument("--border-tolerance", type=float, default=0.03, help="判定截断的贴边阈值")
    parser.add_argument("--large-threshold", type=float, default=0.75, help="判定截断的大框阈值")
    return parser


def _find_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_root = args.source.resolve()
    out_root = args.out.resolve()
    if not source_root.exists():
        print(f"[ERROR] 原始数据集目录不存在: {source_root}")
        return 1

    per_class_limits = {
        0: args.train_per_class + args.val_per_class + args.test_per_class,
        1: args.train_per_class + args.val_per_class + args.test_per_class,
        2: args.train_per_class + args.val_per_class + args.test_per_class,
    }
    items = collect_proxy_items(
        source_root,
        per_class_limits=per_class_limits,
        border_tolerance=args.border_tolerance,
        large_threshold=args.large_threshold,
    )

    by_class: dict[int, list] = defaultdict(list)
    for item in items:
        by_class[item.class_id].append(item)

    for class_id, required_total in per_class_limits.items():
        if len(by_class[class_id]) < required_total:
            max_total = len(by_class[class_id])
            print(f"[ERROR] 类别 {class_id} 可用样本不足，需要 {required_total}，实际 {max_total}")
            print("可尝试减小参数，例如：")
            print(
                "python scripts/build_proxy_gap_dataset.py "
                f"--source {args.source} --out {args.out} "
                "--train-per-class 24 --val-per-class 10 --test-per-class 10"
            )
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

    # 清理旧的代理样本文件，但不删除目录与 data.yaml。
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
    split_plan = [
        ("train", args.train_per_class),
        ("val", args.val_per_class),
        ("test", args.test_per_class),
    ]

    for class_id in sorted(by_class):
        cursor = 0
        for split_name, count in split_plan:
            selected = by_class[class_id][cursor : cursor + count]
            cursor += count
            for item in selected:
                image_target = out_root / "images" / split_name / item.image_path.name
                shutil.copy2(item.image_path, image_target)
                if split_name != "test":
                    rows = parse_label_rows(item.label_path)
                    relabeled = relabel_rows(rows, class_id)
                    label_target = out_root / "labels" / split_name / f"{item.stem}.txt"
                    label_target.write_text("\n".join(relabeled) + "\n", encoding="utf-8")
                    class_counts[class_id] += len(relabeled)
                split_counts[split_name] += 1

    print(f"已构造代理数据集: {out_root}")
    print(f"来源数据集: {source_root}")
    print(
        "规则: 单目标且不贴边=single_gap, "
        "多目标且都不贴边=multi_gap, 所有目标都贴边或超大=truncated, 混合场景跳过。"
    )
    print(f"样本数: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")
    print(
        "训练/验证标签框数量: "
        f"single_gap={class_counts[0]}, multi_gap={class_counts[1]}, truncated={class_counts[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

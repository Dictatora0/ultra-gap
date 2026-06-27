from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StrictItem:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    class_id: int


def parse_label_rows_strict(label_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            rows.append(parts)
    return rows


def _box_features(row: list[str]) -> tuple[float, float, float, float, float, float]:
    _, x_center, y_center, width, height = row
    x = float(x_center)
    y = float(y_center)
    w = float(width)
    h = float(height)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    area = w * h
    aspect = max(w / h, h / w)
    return x1, y1, x2, y2, area, aspect


def categorize_image_strict(rows: list[list[str]]) -> int | None:
    if not rows:
        return None

    features = [_box_features(row) for row in rows]
    clean_center = [
        x1 >= 0.08 and y1 >= 0.08 and x2 <= 0.92 and y2 <= 0.92 and 0.015 <= area <= 0.35 and aspect <= 2.8
        for x1, y1, x2, y2, area, aspect in features
    ]
    trunc_like = [
        (x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99) and area >= 0.03
        for x1, y1, x2, y2, area, _ in features
    ]

    if len(rows) == 1 and clean_center[0]:
        return 0
    if 2 <= len(rows) <= 3 and all(clean_center):
        return 1
    if len(rows) == 1 and trunc_like[0]:
        return 2
    return None


def relabel_rows_strict(rows: list[list[str]], class_id: int) -> list[str]:
    result: list[str] = []
    for row in rows:
        _, x_center, y_center, width, height = row
        result.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return result


def plan_strict_splits(
    items: list[StrictItem],
    *,
    use_all_available: bool,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> dict[str, list[StrictItem]]:
    if use_all_available:
        train_items = [item for item in items if item.split == "train"]
        val_items = [item for item in items if item.split == "val"]
        # 没有额外 test 标签时，用严格合格的验证图像作为 test 预测输入示例。
        test_items = list(val_items)
        return {"train": train_items, "val": val_items, "test": test_items}

    by_class: dict[int, list[StrictItem]] = defaultdict(list)
    for item in items:
        by_class[item.class_id].append(item)

    split_plan = {"train": [], "val": [], "test": []}
    for class_id in sorted(by_class):
        cursor = 0
        for split_name, count in [("train", train_per_class), ("val", val_per_class), ("test", test_per_class)]:
            selected = by_class[class_id][cursor : cursor + count]
            split_plan[split_name].extend(selected)
            cursor += count
    return split_plan


def collect_strict_items(source_root: Path, *, per_class_limits: dict[int, int]) -> list[StrictItem]:
    buckets: dict[int, list[StrictItem]] = defaultdict(list)
    for split in ("train", "val"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.png"
            if not image_path.exists():
                continue
            rows = parse_label_rows_strict(label_path)
            class_id = categorize_image_strict(rows)
            if class_id is None:
                continue
            if len(buckets[class_id]) >= per_class_limits.get(class_id, 0):
                continue
            buckets[class_id].append(
                StrictItem(
                    split=split,
                    stem=label_path.stem,
                    image_path=image_path,
                    label_path=label_path,
                    class_id=class_id,
                )
            )
    items: list[StrictItem] = []
    for class_id in sorted(per_class_limits):
        items.extend(buckets[class_id])
    return items

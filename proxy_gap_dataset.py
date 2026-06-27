from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProxyItem:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    class_id: int


def parse_label_rows(label_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            rows.append(parts)
    return rows


def _is_truncated_row(row: list[str], border_tolerance: float, large_threshold: float) -> bool:
    _, x_center, y_center, width, height = row
    x = float(x_center)
    y = float(y_center)
    w = float(width)
    h = float(height)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    touches_border = x1 <= border_tolerance or y1 <= border_tolerance or x2 >= 1 - border_tolerance or y2 >= 1 - border_tolerance
    is_large = w >= large_threshold or h >= large_threshold
    return touches_border or is_large


def categorize_image(
    rows: list[list[str]],
    *,
    border_tolerance: float = 0.03,
    large_threshold: float = 0.75,
) -> int | None:
    if not rows:
        return None

    trunc_flags = [_is_truncated_row(row, border_tolerance, large_threshold) for row in rows]
    if any(trunc_flags) and not all(trunc_flags):
        return None
    if all(trunc_flags):
        return 2
    if len(rows) >= 2:
        return 1
    return 0


def relabel_rows(rows: list[list[str]], class_id: int) -> list[str]:
    relabeled: list[str] = []
    for row in rows:
        _, x_center, y_center, width, height = row
        relabeled.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return relabeled


def summarize_available_classes(
    row_sets: list[list[list[str]]],
    *,
    border_tolerance: float = 0.03,
    large_threshold: float = 0.75,
) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for rows in row_sets:
        category = categorize_image(rows, border_tolerance=border_tolerance, large_threshold=large_threshold)
        if category is not None:
            counts[category] += 1
    return counts


def collect_proxy_items(
    source_root: Path,
    *,
    per_class_limits: dict[int, int],
    border_tolerance: float = 0.03,
    large_threshold: float = 0.75,
) -> list[ProxyItem]:
    buckets: dict[int, list[ProxyItem]] = defaultdict(list)
    for split in ("train", "val"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_dir / f"{label_path.stem}.png"
            if not image_path.exists():
                continue
            rows = parse_label_rows(label_path)
            category = categorize_image(rows, border_tolerance=border_tolerance, large_threshold=large_threshold)
            if category is None:
                continue
            if len(buckets[category]) >= per_class_limits.get(category, 0):
                continue
            buckets[category].append(
                ProxyItem(
                    split=split,
                    stem=label_path.stem,
                    image_path=image_path,
                    label_path=label_path,
                    class_id=category,
                )
            )

    items: list[ProxyItem] = []
    for class_id in sorted(per_class_limits):
        items.extend(buckets[class_id])
    return items

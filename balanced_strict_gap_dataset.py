from __future__ import annotations

from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    from strict_gap_dataset import StrictItem
else:
    from .strict_gap_dataset import StrictItem


def make_augmented_stem(stem: str, variant: str) -> str:
    return f"{stem}__aug_{variant}"


def build_balanced_training_plan(
    items: list[StrictItem],
    *,
    target_per_class: int,
) -> dict[int, list[dict[str, object]]]:
    by_class: dict[int, list[StrictItem]] = defaultdict(list)
    for item in items:
        if item.split == "train":
            by_class[item.class_id].append(item)

    variants = ["orig", "flip", "bright", "flip_bright"]
    plan: dict[int, list[dict[str, object]]] = {}
    for class_id, class_items in by_class.items():
        if not class_items:
            plan[class_id] = []
            continue
        class_plan: list[dict[str, object]] = []
        cursor = 0
        while len(class_plan) < target_per_class:
            source = class_items[cursor % len(class_items)]
            round_index = cursor // len(class_items)
            variant = variants[min(round_index, len(variants) - 1)]
            class_plan.append({"source": source, "variant": variant})
            cursor += 1
        plan[class_id] = class_plan
    return plan

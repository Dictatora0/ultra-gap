import unittest
from pathlib import Path

from gap_detection_ultralytics.strict_gap_dataset import (
    StrictItem,
    categorize_image_strict,
    plan_strict_splits,
    relabel_rows_strict,
)


class CategorizeImageStrictTests(unittest.TestCase):
    def test_keeps_single_clean_center_box(self) -> None:
        category = categorize_image_strict([["1", "0.50", "0.50", "0.20", "0.20"]])
        self.assertEqual(category, 0)

    def test_rejects_single_box_near_edge(self) -> None:
        category = categorize_image_strict([["1", "0.93", "0.50", "0.12", "0.12"]])
        self.assertIsNone(category)

    def test_keeps_two_clean_center_boxes_as_multi(self) -> None:
        category = categorize_image_strict(
            [
                ["0", "0.35", "0.40", "0.16", "0.16"],
                ["3", "0.68", "0.58", "0.14", "0.14"],
            ]
        )
        self.assertEqual(category, 1)

    def test_rejects_multi_box_with_too_many_targets(self) -> None:
        category = categorize_image_strict(
            [
                ["0", "0.20", "0.30", "0.10", "0.10"],
                ["1", "0.40", "0.40", "0.10", "0.10"],
                ["2", "0.60", "0.50", "0.10", "0.10"],
                ["3", "0.80", "0.60", "0.10", "0.10"],
            ]
        )
        self.assertIsNone(category)

    def test_keeps_single_large_border_box_as_truncated(self) -> None:
        category = categorize_image_strict([["2", "0.88", "0.50", "0.22", "0.30"]])
        self.assertEqual(category, 2)

    def test_rejects_truncated_candidate_when_box_too_small(self) -> None:
        category = categorize_image_strict([["2", "0.97", "0.08", "0.03", "0.03"]])
        self.assertIsNone(category)


class RelabelRowsStrictTests(unittest.TestCase):
    def test_rewrites_class_ids(self) -> None:
        relabeled = relabel_rows_strict([["3", "0.50", "0.50", "0.20", "0.20"]], class_id=2)
        self.assertEqual(relabeled, ["2 0.50 0.50 0.20 0.20"])


class PlanStrictSplitsTests(unittest.TestCase):
    def test_use_all_available_preserves_source_train_val_and_mirrors_val_to_test(self) -> None:
        items = [
            StrictItem("train", "train_a", Path("train_a.png"), Path("train_a.txt"), 0),
            StrictItem("train", "train_b", Path("train_b.png"), Path("train_b.txt"), 1),
            StrictItem("val", "val_a", Path("val_a.png"), Path("val_a.txt"), 0),
            StrictItem("val", "val_b", Path("val_b.png"), Path("val_b.txt"), 2),
        ]

        split_plan = plan_strict_splits(
            items,
            use_all_available=True,
            train_per_class=1,
            val_per_class=1,
            test_per_class=1,
        )

        self.assertEqual([item.stem for item in split_plan["train"]], ["train_a", "train_b"])
        self.assertEqual([item.stem for item in split_plan["val"]], ["val_a", "val_b"])
        self.assertEqual([item.stem for item in split_plan["test"]], ["val_a", "val_b"])


if __name__ == "__main__":
    unittest.main()

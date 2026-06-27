import unittest
from pathlib import Path

from gap_detection_ultralytics.strict_gap_dataset import StrictItem
from gap_detection_ultralytics.balanced_strict_gap_dataset import (
    build_balanced_training_plan,
    make_augmented_stem,
)


class MakeAugmentedStemTests(unittest.TestCase):
    def test_appends_variant_suffix(self) -> None:
        self.assertEqual(make_augmented_stem("road1", "flip"), "road1__aug_flip")


class BuildBalancedTrainingPlanTests(unittest.TestCase):
    def test_upsamples_only_minority_classes_to_target_count(self) -> None:
        items = [
            StrictItem("train", "a0", Path("a0.png"), Path("a0.txt"), 0),
            StrictItem("train", "a1", Path("a1.png"), Path("a1.txt"), 0),
            StrictItem("train", "b0", Path("b0.png"), Path("b0.txt"), 1),
            StrictItem("train", "c0", Path("c0.png"), Path("c0.txt"), 2),
        ]

        plan = build_balanced_training_plan(items, target_per_class=2)

        self.assertEqual(len(plan[0]), 2)
        self.assertEqual(len(plan[1]), 2)
        self.assertEqual(len(plan[2]), 2)
        self.assertEqual(plan[0][0]["source"].stem, "a0")
        self.assertEqual(plan[1][1]["variant"], "flip")
        self.assertEqual(plan[2][1]["variant"], "flip")


if __name__ == "__main__":
    unittest.main()

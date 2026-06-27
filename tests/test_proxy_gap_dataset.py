import unittest

from gap_detection_ultralytics.proxy_gap_dataset import categorize_image, relabel_rows, summarize_available_classes


class CategorizeImageTests(unittest.TestCase):
    def test_single_center_box_maps_to_single_gap(self) -> None:
        category = categorize_image([["3", "0.50", "0.50", "0.20", "0.20"]])
        self.assertEqual(category, 0)

    def test_multiple_center_boxes_map_to_multi_gap(self) -> None:
        category = categorize_image(
            [
                ["1", "0.30", "0.40", "0.10", "0.12"],
                ["0", "0.70", "0.60", "0.11", "0.10"],
            ]
        )
        self.assertEqual(category, 1)

    def test_border_touching_box_maps_to_truncated(self) -> None:
        category = categorize_image([["2", "0.92", "0.50", "0.20", "0.20"]], border_tolerance=0.03)
        self.assertEqual(category, 2)

    def test_mixed_truncated_and_non_truncated_image_is_skipped(self) -> None:
        category = categorize_image(
            [
                ["2", "0.92", "0.50", "0.20", "0.20"],
                ["0", "0.30", "0.40", "0.10", "0.10"],
            ],
            border_tolerance=0.03,
        )
        self.assertIsNone(category)


class RelabelRowsTests(unittest.TestCase):
    def test_rewrites_all_rows_to_target_class(self) -> None:
        relabeled = relabel_rows(
            [
                ["3", "0.50", "0.50", "0.20", "0.20"],
                ["1", "0.70", "0.60", "0.10", "0.12"],
            ],
            class_id=1,
        )
        self.assertEqual(
            relabeled,
            [
                "1 0.50 0.50 0.20 0.20",
                "1 0.70 0.60 0.10 0.12",
            ],
        )


class SummarizeAvailableClassesTests(unittest.TestCase):
    def test_reports_clean_pool_sizes(self) -> None:
        summary = summarize_available_classes(
            [
                [["3", "0.50", "0.50", "0.20", "0.20"]],
                [["1", "0.30", "0.40", "0.10", "0.12"], ["0", "0.70", "0.60", "0.11", "0.10"]],
                [["2", "0.92", "0.50", "0.20", "0.20"]],
                [["2", "0.92", "0.50", "0.20", "0.20"], ["0", "0.30", "0.40", "0.10", "0.10"]],
            ]
        )
        self.assertEqual(summary, {0: 1, 1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from gap_detection_ultralytics.gap_core import (
    DatasetCheckSummary,
    collect_dataset_summary,
    gather_image_label_pairs,
    list_export_candidates,
    resolve_dataset_root,
    temporary_ultralytics_data_config,
)
from gap_detection_ultralytics.scripts.predict import resolve_save_location
from gap_detection_ultralytics.scripts.val import resolve_val_save_location


class CollectDatasetSummaryTests(unittest.TestCase):
    def test_reports_missing_label_and_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images_train = root / "images" / "train"
            images_val = root / "images" / "val"
            labels_train = root / "labels" / "train"
            labels_val = root / "labels" / "val"
            for path in [images_train, images_val, labels_train, labels_val]:
                path.mkdir(parents=True, exist_ok=True)

            (images_train / "sample1.jpg").write_bytes(b"img")
            (images_train / "sample2.png").write_bytes(b"img")
            (images_val / "val1.jpg").write_bytes(b"img")

            (labels_train / "sample1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (labels_val / "val1.txt").write_text("3 0.5 0.5 0.2 1.2\n", encoding="utf-8")

            summary = collect_dataset_summary(root)

        self.assertIsInstance(summary, DatasetCheckSummary)
        self.assertEqual(summary.image_counts["train"], 2)
        self.assertEqual(summary.image_counts["val"], 1)
        self.assertEqual(summary.label_counts["train"], 1)
        self.assertEqual(summary.class_counts[0], 1)
        self.assertIn("sample2", "\n".join(summary.warnings))
        self.assertIn("val1.txt", "\n".join(summary.errors))


class GatherImageLabelPairsTests(unittest.TestCase):
    def test_pairs_only_images_with_matching_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            (images_dir / "a.jpg").write_bytes(b"img")
            (images_dir / "b.png").write_bytes(b"img")
            (labels_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            pairs, warnings = gather_image_label_pairs(images_dir, labels_dir)

        self.assertEqual([pair.stem for pair in pairs], ["a"])
        self.assertIn("b", "\n".join(warnings))


class DatasetPathResolutionTests(unittest.TestCase):
    def test_resolve_dataset_root_uses_yaml_parent_when_tail_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data" / "gap_dataset"
            root.mkdir(parents=True)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: data/gap_dataset\ntrain: images/train\nval: images/val\n",
                encoding="utf-8",
            )

            resolved = resolve_dataset_root(yaml_path)

        self.assertEqual(resolved, root.resolve())

    def test_temporary_ultralytics_data_config_writes_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data" / "gap_dataset"
            root.mkdir(parents=True)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: data/gap_dataset\ntrain: images/train\nval: images/val\n",
                encoding="utf-8",
            )

            with temporary_ultralytics_data_config(yaml_path) as temp_yaml:
                content = Path(temp_yaml).read_text(encoding="utf-8")

        self.assertIn(f"path: {root.resolve()}", content)


class ExportCandidatesTests(unittest.TestCase):
    def test_marks_missing_optional_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"weights")
            (run_dir / "results.png").write_bytes(b"image")

            required, optional_missing = list_export_candidates(run_dir)

        required_names = {source.name for source, _ in required}
        self.assertIn("best.pt", required_names)
        self.assertIn("results.png", required_names)
        self.assertIn("confusion_matrix.png", optional_missing)

    def test_supports_box_curve_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"weights")
            (run_dir / "results.png").write_bytes(b"image")
            (run_dir / "confusion_matrix.png").write_bytes(b"image")
            (run_dir / "BoxPR_curve.png").write_bytes(b"image")
            (run_dir / "BoxF1_curve.png").write_bytes(b"image")
            (run_dir / "BoxP_curve.png").write_bytes(b"image")
            (run_dir / "BoxR_curve.png").write_bytes(b"image")

            available, missing = list_export_candidates(run_dir)

        destinations = {destination.name for _, destination in available}
        self.assertIn("PR_curve.png", destinations)
        self.assertIn("F1_curve.png", destinations)
        self.assertIn("P_curve.png", destinations)
        self.assertIn("R_curve.png", destinations)
        self.assertNotIn("PR_curve.png", missing)
        self.assertNotIn("F1_curve.png", missing)
        self.assertNotIn("P_curve.png", missing)
        self.assertNotIn("R_curve.png", missing)


class PredictScriptTests(unittest.TestCase):
    def test_resolve_save_location_returns_absolute_project_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "outputs" / "predict_examples"
            resolved_dir, project_dir, run_name = resolve_save_location(save_dir)

        self.assertTrue(resolved_dir.is_absolute())
        self.assertEqual(project_dir, resolved_dir.parent)
        self.assertEqual(run_name, "predict_examples")


class ValScriptTests(unittest.TestCase):
    def test_resolve_val_save_location_uses_absolute_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir, project_dir, run_name = resolve_val_save_location(Path(tmp_dir) / "outputs" / "val" / "gap_val")

        self.assertTrue(save_dir.is_absolute())
        self.assertEqual(project_dir, save_dir.parent)
        self.assertEqual(run_name, "gap_val")


if __name__ == "__main__":
    unittest.main()

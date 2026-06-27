import os
import tempfile
import unittest
from pathlib import Path

from gap_detection_ultralytics.scripts.check_dataset import resolve_dataset_root


class ResolveDatasetRootTests(unittest.TestCase):
    def test_resolves_project_relative_path_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            dataset_root = project_root / "data" / "gap_dataset"
            (dataset_root / "images" / "train").mkdir(parents=True)
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "labels" / "train").mkdir(parents=True)
            (dataset_root / "labels" / "val").mkdir(parents=True)
            data_yaml = dataset_root / "data.yaml"
            data_yaml.write_text("path: data/gap_dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")

            original_cwd = Path.cwd()
            os.chdir(project_root)
            try:
                resolved = resolve_dataset_root(data_yaml)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(resolved, dataset_root.resolve())

    def test_falls_back_to_yaml_parent_when_path_is_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            (dataset_root / "images" / "train").mkdir(parents=True)
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "labels" / "train").mkdir(parents=True)
            (dataset_root / "labels" / "val").mkdir(parents=True)
            data_yaml = dataset_root / "roadsign.yaml"
            data_yaml.write_text("path: 替换为自己的数据集地址\ntrain: images/train\nval: images/val\n", encoding="utf-8")

            resolved = resolve_dataset_root(data_yaml)

        self.assertEqual(resolved, dataset_root.resolve())

    def test_uses_yaml_parent_when_path_suffix_matches_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            project_root = workspace_root / "gap_detection_ultralytics"
            dataset_root = project_root / "data" / "gap_dataset"
            (dataset_root / "images" / "train").mkdir(parents=True)
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "labels" / "train").mkdir(parents=True)
            (dataset_root / "labels" / "val").mkdir(parents=True)
            data_yaml = dataset_root / "data.yaml"
            data_yaml.write_text("path: data/gap_dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")

            original_cwd = Path.cwd()
            os.chdir(workspace_root)
            try:
                resolved = resolve_dataset_root(data_yaml)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(resolved, dataset_root.resolve())


if __name__ == "__main__":
    unittest.main()

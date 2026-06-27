import unittest
from pathlib import Path


def compute_output_root(base_dir: Path, out_arg: Path) -> Path:
    if out_arg.is_absolute():
        return out_arg
    return (base_dir / out_arg).resolve()


class BuildStrictGapDatasetPathingTests(unittest.TestCase):
    def test_relative_out_path_stays_inside_project(self) -> None:
        project_root = Path("/tmp/project")
        out_root = compute_output_root(project_root, Path("data/gap_dataset_strict"))
        self.assertEqual(out_root.relative_to(project_root.resolve()), Path("data/gap_dataset_strict"))


if __name__ == "__main__":
    unittest.main()

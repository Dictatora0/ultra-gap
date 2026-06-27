from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "single_gap",
    1: "multi_gap",
    2: "truncated",
}


@dataclass
class DatasetCheckSummary:
    image_counts: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    label_counts: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})
    class_counts: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    label_path: Path

    @property
    def stem(self) -> str:
        return self.image_path.stem


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def gather_image_label_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[ImageLabelPair], list[str]]:
    pairs: list[ImageLabelPair] = []
    warnings: list[str] = []

    for image_path in iter_image_files(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path))
        else:
            warnings.append(f"缺少标签文件: {image_path.name} -> {label_path.name}")

    return pairs, warnings


def _validate_label_line(
    line: str,
    *,
    label_path: Path,
    line_number: int,
    class_counts: dict[int, int],
) -> list[str]:
    errors: list[str] = []
    parts = line.strip().split()
    if len(parts) != 5:
        return [f"{label_path.name} 第 {line_number} 行格式错误，应为 5 列，实际为 {len(parts)} 列。"]

    try:
        class_id = int(parts[0])
    except ValueError:
        return [f"{label_path.name} 第 {line_number} 行 class_id 不是整数。"]

    if class_id not in CLASS_NAMES:
        errors.append(f"{label_path.name} 第 {line_number} 行 class_id={class_id} 超出范围，仅支持 0/1/2。")

    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return [f"{label_path.name} 第 {line_number} 行坐标不是合法浮点数。"]

    for value_index, coord in enumerate(coords, start=1):
        if not 0.0 <= coord <= 1.0:
            errors.append(
                f"{label_path.name} 第 {line_number} 行第 {value_index + 1} 列坐标={coord} 不在 [0, 1] 范围内。"
            )

    if not errors and class_id in CLASS_NAMES:
        class_counts[class_id] += 1
    return errors


def collect_dataset_summary(dataset_root: Path) -> DatasetCheckSummary:
    summary = DatasetCheckSummary()
    required_dirs = {
        "train_images": dataset_root / "images" / "train",
        "val_images": dataset_root / "images" / "val",
        "train_labels": dataset_root / "labels" / "train",
        "val_labels": dataset_root / "labels" / "val",
    }

    for name, path in required_dirs.items():
        if not path.exists():
            summary.errors.append(f"缺少目录: {name} -> {path}")

    if summary.errors:
        return summary

    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        image_files = iter_image_files(images_dir)
        label_files = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())

        summary.image_counts[split] = len(image_files)
        summary.label_counts[split] = len(label_files)

        pairs, warnings = gather_image_label_pairs(images_dir, labels_dir)
        summary.warnings.extend(f"[{split}] {warning}" for warning in warnings)

        image_stems = {path.stem for path in image_files}
        for label_path in label_files:
            if label_path.stem not in image_stems:
                summary.warnings.append(f"[{split}] 存在孤立标签文件: {label_path.name}")

            content = label_path.read_text(encoding="utf-8").splitlines()
            if not content:
                summary.warnings.append(f"[{split}] 空标签文件: {label_path.name}")
                continue

            for line_number, line in enumerate(content, start=1):
                if not line.strip():
                    summary.warnings.append(f"[{split}] {label_path.name} 第 {line_number} 行为空行。")
                    continue
                summary.errors.extend(
                    f"[{split}] {error}"
                    for error in _validate_label_line(
                        line,
                        label_path=label_path,
                        line_number=line_number,
                        class_counts=summary.class_counts,
                    )
                )

    return summary


def format_summary_lines(summary: DatasetCheckSummary) -> list[str]:
    lines = [
        "数据集检查结果",
        f"训练集图片数: {summary.image_counts['train']}",
        f"训练集标签数: {summary.label_counts['train']}",
        f"验证集图片数: {summary.image_counts['val']}",
        f"验证集标签数: {summary.label_counts['val']}",
        "类别分布:",
    ]
    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"  {class_id} {class_name}: {summary.class_counts[class_id]}")

    if summary.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in summary.warnings)
    if summary.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in summary.errors)

    lines.append("检查结论: 通过" if summary.is_valid else "检查结论: 存在问题，请根据以上提示修正。")
    return lines


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _looks_like_placeholder(raw_path: str) -> bool:
    return "替换" in raw_path or "自己" in raw_path or raw_path.strip() in {"", "."}


def _path_matches_tail(candidate_root: Path, relative_path: Path) -> bool:
    relative_parts = relative_path.parts
    if len(relative_parts) > len(candidate_root.parts):
        return False
    return tuple(candidate_root.parts[-len(relative_parts) :]) == relative_parts


def load_data_config(data_arg: Path) -> dict:
    if not data_arg.exists():
        raise FileNotFoundError(f"未找到数据配置文件或数据目录: {data_arg}")
    if data_arg.is_dir():
        return {"path": str(data_arg.resolve())}

    with data_arg.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"data.yaml 内容格式错误，应为字典结构: {data_arg}")
    return config


def resolve_dataset_root(data_arg: Path) -> Path:
    if data_arg.is_dir():
        return data_arg.resolve()

    config = load_data_config(data_arg)
    raw_path = str(config.get("path", "."))
    if _looks_like_placeholder(raw_path):
        return data_arg.parent.resolve()

    base_path = Path(raw_path)
    if base_path.is_absolute():
        return base_path

    if _path_matches_tail(data_arg.parent.resolve(), base_path):
        return data_arg.parent.resolve()

    cwd_candidate = (Path.cwd() / base_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    yaml_parent_candidate = (data_arg.parent / base_path).resolve()
    if yaml_parent_candidate.exists():
        return yaml_parent_candidate

    return yaml_parent_candidate


@contextmanager
def temporary_ultralytics_data_config(data_arg: Path) -> Iterator[str]:
    """将相对 path 的 data.yaml 转换为 Ultralytics 可稳定识别的绝对路径配置。"""
    if data_arg.is_dir():
        yield str(data_arg.resolve())
        return

    config = load_data_config(data_arg)
    normalized_config = dict(config)
    normalized_config["path"] = str(resolve_dataset_root(data_arg))

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(normalized_config, handle, allow_unicode=True, sort_keys=False)
        temp_path = Path(handle.name)

    try:
        yield str(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def resolve_output_location(save_dir: Path) -> tuple[Path, Path, str]:
    """将保存目录解析为绝对路径，避免 Ultralytics 按权重目录错误拼接。"""
    resolved_dir = save_dir.resolve()
    return resolved_dir, resolved_dir.parent, resolved_dir.name


def list_export_candidates(run_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    candidates = [
        ((run_dir / "weights" / "best.pt",), Path("best.pt")),
        ((run_dir / "results.png",), Path("results.png")),
        ((run_dir / "confusion_matrix.png",), Path("confusion_matrix.png")),
        ((run_dir / "PR_curve.png", run_dir / "BoxPR_curve.png"), Path("PR_curve.png")),
        ((run_dir / "F1_curve.png", run_dir / "BoxF1_curve.png"), Path("F1_curve.png")),
        ((run_dir / "P_curve.png", run_dir / "BoxP_curve.png"), Path("P_curve.png")),
        ((run_dir / "R_curve.png", run_dir / "BoxR_curve.png"), Path("R_curve.png")),
    ]
    available: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for source_options, destination in candidates:
        source = next((path for path in source_options if path.exists()), None)
        if source is not None:
            available.append((source, destination))
            continue
        missing.append(destination.name)
    return available, missing

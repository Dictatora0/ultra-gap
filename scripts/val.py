from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import resolve_output_location, temporary_ultralytics_data_config
else:
    from ..gap_core import resolve_output_location, temporary_ultralytics_data_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证训练后的 YOLO 缺口识别模型。")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/gap_train/weights/best.pt"),
        help="待验证的权重文件",
    )
    parser.add_argument("--data", type=Path, default=Path("data/gap_dataset/data.yaml"), help="data.yaml 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="验证图像尺寸")
    parser.add_argument("--device", type=str, default="cpu", help="验证设备，例如 cpu 或 0")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/val/gap_val"),
        help="验证结果保存目录",
    )
    return parser


def resolve_val_save_location(save_dir: Path) -> tuple[Path, Path, str]:
    return resolve_output_location(save_dir)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"[ERROR] 权重文件不存在: {args.weights}")
        print("请先完成训练，或通过 --weights 指向正确的 best.pt。")
        return 1
    if not args.data.exists():
        print(f"[ERROR] data.yaml 不存在: {args.data}")
        return 1

    save_dir, project_dir, run_name = resolve_val_save_location(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    with temporary_ultralytics_data_config(args.data) as normalized_data:
        metrics = model.val(
            data=normalized_data,
            imgsz=args.imgsz,
            device=args.device,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
        )

    box = metrics.box
    print("验证完成，关键指标如下:")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    print(f"结果目录: {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import CLASS_NAMES, ensure_directories, resolve_output_location
else:
    from ..gap_core import CLASS_NAMES, ensure_directories, resolve_output_location


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对图片或文件夹执行缺口识别预测。")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/gap_train/weights/best.pt"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/gap_dataset/images/test"),
        help="待预测的图片路径或目录",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="预测图像尺寸")
    parser.add_argument("--device", type=str, default="cpu", help="预测设备，例如 cpu 或 0")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/predict_examples"),
        help="保存带框结果图的目录",
    )
    return parser


def resolve_save_location(save_dir: Path) -> tuple[Path, Path, str]:
    return resolve_output_location(save_dir)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"[ERROR] 模型权重不存在: {args.weights}")
        return 1
    if not args.source.exists():
        print(f"[ERROR] 预测输入不存在: {args.source}")
        return 1

    save_dir, project_dir, run_name = resolve_save_location(args.save_dir)
    ensure_directories([save_dir])
    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    print(f"预测完成，结果图片保存在: {save_dir}")
    for result in results:
        print(f"\n图片: {Path(result.path).name}")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("  未检测到目标。")
            continue

        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())
            x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[index].tolist()]
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            print(
                f"  检测{index + 1}: class={class_name}, conf={confidence:.4f}, "
                f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

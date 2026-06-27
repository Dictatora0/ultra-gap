from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import temporary_ultralytics_data_config
else:
    from ..gap_core import temporary_ultralytics_data_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Ultralytics YOLO 训练缺口识别模型。")
    parser.add_argument("--data", type=Path, default=Path("data/gap_dataset/data.yaml"), help="data.yaml 路径")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="预训练模型名称或权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批大小")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备，例如 cpu、0、0,1")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="训练结果根目录")
    parser.add_argument("--name", type=str, default="gap_train", help="本次训练名称")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] 未找到数据配置文件: {args.data}")
        print("请先准备 data/gap_dataset/data.yaml，或通过 --data 指定正确路径。")
        return 1

    print("开始训练 YOLO 模型。")
    print(f"data={args.data}")
    print(f"model={args.model}")
    print(f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")

    # 课程实验中建议使用轻量模型，CPU 环境可先跑通流程后再切换更高配置。
    model = YOLO(args.model)
    with temporary_ultralytics_data_config(args.data) as normalized_data:
        results = model.train(
            data=normalized_data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project),
            name=args.name,
        )

    save_dir = Path(results.save_dir)
    print("训练完成，关键文件位置如下:")
    print(f"best.pt: {save_dir / 'weights' / 'best.pt'}")
    print(f"last.pt: {save_dir / 'weights' / 'last.pt'}")
    print(f"results.png: {save_dir / 'results.png'}")
    print(f"confusion_matrix.png: {save_dir / 'confusion_matrix.png'}")
    print(f"训练目录: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

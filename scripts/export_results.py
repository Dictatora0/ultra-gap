from __future__ import annotations

import argparse
import shutil
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gap_core import ensure_directories, list_export_candidates
else:
    from ..gap_core import ensure_directories, list_export_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="整理训练结果与预测示例，便于课程实验提交。")
    parser.add_argument("--run", type=Path, default=Path("runs/detect/gap_train"), help="训练输出目录")
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("outputs/predict_examples"),
        help="预测示例目录",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/submit"), help="提交材料输出目录")
    return parser


def write_submit_readme(out_dir: Path, run_dir: Path, pred_dir: Path) -> None:
    content = "\n".join(
        [
            "实验名称：基于 Ultralytics 的单缺口、多缺口和截取识别实验",
            "类别说明：",
            "0 single_gap",
            "1 multi_gap",
            "2 truncated",
            "",
            "训练命令：",
            "python scripts/train.py --data data/gap_dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --device cpu",
            "",
            "验证命令：",
            "python scripts/val.py --weights runs/detect/gap_train/weights/best.pt --data data/gap_dataset/data.yaml",
            "",
            "预测命令：",
            "python scripts/predict.py --weights runs/detect/gap_train/weights/best.pt --source data/gap_dataset/images/test --conf 0.25",
            "",
            "提交文件说明：",
            f"1. best.pt：训练得到的最优模型权重，默认来源 {run_dir / 'weights' / 'best.pt'}",
            "2. results.png：训练过程指标曲线图。",
            "3. confusion_matrix.png：混淆矩阵图。",
            "4. PR/F1/P/R 曲线：若训练目录存在则一并复制。",
            f"5. predict_examples/：预测示例图片，默认来源 {pred_dir}",
            "6. 本说明文件：用于群内提交时快速说明实验内容与命令。",
        ]
    )
    (out_dir / "README_submit.txt").write_text(content, encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ensure_directories([args.out, args.out / "predict_examples"])
    if not args.run.exists():
        print(f"[WARNING] 训练结果目录不存在: {args.run}")

    copied_any = False
    available, missing = list_export_candidates(args.run)
    for source, relative_target in available:
        target = args.out / relative_target
        shutil.copy2(source, target)
        copied_any = True
        print(f"已复制: {source} -> {target}")
    for name in missing:
        print(f"[WARNING] 结果文件不存在，已跳过: {name}")

    if args.pred.exists():
        for item in args.pred.iterdir():
            if item.is_file():
                shutil.copy2(item, args.out / "predict_examples" / item.name)
                copied_any = True
        print(f"已整理预测示例到: {args.out / 'predict_examples'}")
    else:
        print(f"[WARNING] 预测示例目录不存在，已跳过: {args.pred}")

    write_submit_readme(args.out, args.run, args.pred)
    print(f"已生成提交说明: {args.out / 'README_submit.txt'}")
    if not copied_any:
        print("[WARNING] 当前未复制到实际结果文件，请先完成训练和预测后重新执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

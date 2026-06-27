实验名称：基于 Ultralytics 的单缺口、多缺口和截取识别实验
类别说明：
0 single_gap
1 multi_gap
2 truncated

训练命令：
python scripts/train.py --data data/gap_dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --device cpu

验证命令：
python scripts/val.py --weights runs/detect/gap_train/weights/best.pt --data data/gap_dataset/data.yaml

预测命令：
python scripts/predict.py --weights runs/detect/gap_train/weights/best.pt --source data/gap_dataset/images/test --conf 0.25

提交文件说明：
1. best.pt：训练得到的最优模型权重，默认来源 /Users/lifulin/Desktop/pocketfit/runs/detect/koala-raven/example-project/train/weights/best.pt
2. results.png：训练过程指标曲线图。
3. confusion_matrix.png：混淆矩阵图。
4. PR/F1/P/R 曲线：若训练目录存在则一并复制。
5. predict_examples/：预测示例图片，默认来源 gap_detection_ultralytics/outputs/predict_examples
6. 本说明文件：用于群内提交时快速说明实验内容与命令。
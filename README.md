# 基于 Ultralytics 的单缺口、多缺口和截取识别实验

本工程面向课程实验提交，使用 Ultralytics YOLO 完成三类目标检测任务：`single_gap`、`multi_gap`、`truncated`。项目已经提供数据检查、数据划分、训练、验证、预测和结果整理脚本，适合直接按照实验流程执行。

当前仓库还额外提供了一个“代理重标注”流程：可以从现有检测数据中抽取一部分样本，按规则映射成三类课程实验数据，用于快速跑通完整实验链路。该代理数据集适合课程演示和流程验证，但其语义并不等同于真实工业场景中的缺口/截取标注，实验报告中应明确说明数据来源和转换规则。

## 1. 实验简介

实验目标是训练一个 YOLO 检测模型，对图像中的三类目标进行识别：

- `0 single_gap`：单缺口
- `1 multi_gap`：多缺口
- `2 truncated`：截取/截断

推荐默认模型为 `yolo11n.pt`。如果当前环境无法下载该模型，或课程机器上的 Ultralytics 版本不支持，也可以改用 `yolov8n.pt`。

## 2. 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

如果你在 CPU 环境中仅做流程演示，可以保持默认参数。若有 NVIDIA GPU，可通过 `--device 0` 指定 GPU 训练。

## 3. 数据集目录格式

项目默认使用以下目录结构：

```text
gap_detection_ultralytics/
├── data/
│   └── gap_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml
```

如果你已经有现成数据，不需要移动到仓库根目录。只需将训练图片放入 `data/gap_dataset/images/train`，验证图片放入 `data/gap_dataset/images/val`，并将同名标签文件放入 `data/gap_dataset/labels/train` 和 `data/gap_dataset/labels/val` 即可。测试集图片可放入 `data/gap_dataset/images/test`。

如果现有数据还未划分，可先使用 `scripts/split_dataset.py` 从原始目录自动划分。

如果你当前只有其他目标检测数据，而没有现成的 `single_gap / multi_gap / truncated` 标注，也可以使用仓库提供的代理构造脚本：

```bash
python scripts/build_proxy_gap_dataset.py --source ../dataset --out data/gap_dataset --train-per-class 24 --val-per-class 10 --test-per-class 10
```

默认代理规则如下：

- 单目标且边界框不贴边：映射为 `single_gap`
- 多目标且所有边界框都不贴边：映射为 `multi_gap`
- 所有目标都贴边或边界框过大：映射为 `truncated`
- 同时包含贴边和非贴边目标的混合场景：跳过，不纳入代理数据集

## 4. 三类标签说明

类别定义如下：

```text
0 single_gap
1 multi_gap
2 truncated
```

`data/gap_dataset/data.yaml` 内容如下：

```yaml
path: data/gap_dataset
train: images/train
val: images/val
test: images/test

names:
  0: single_gap
  1: multi_gap
  2: truncated
```

## 5. 标注格式说明

本实验使用 YOLO Detection 标注格式。每张图片对应一个同名 `.txt` 文件，每一行为一个目标：

```text
class_id x_center y_center width height
```

说明：

- `class_id` 只能是 `0`、`1`、`2`
- 坐标均为归一化后的相对值
- `x_center`、`y_center`、`width`、`height` 都必须在 `0` 到 `1` 之间

示例：

```text
0 0.532 0.478 0.210 0.165
1 0.314 0.622 0.180 0.240
```

## 6. 数据集检查命令

在训练前先检查目录和标注格式：

```bash
python scripts/check_dataset.py --data data/gap_dataset/data.yaml
```

脚本会检查：

- `images/train`、`images/val`、`labels/train`、`labels/val` 是否存在
- 每张图片是否有同名标签文件
- 标签文件是否符合 YOLO 格式
- 类别 ID 是否为 `0/1/2`
- 坐标是否在 `[0, 1]`
- 训练集与验证集的样本数量和类别分布

## 7. 数据划分命令

如果原始数据还没有划分，可以执行：

```bash
python scripts/split_dataset.py --images /path/to/raw_images --labels /path/to/raw_labels --out data/gap_dataset --val-ratio 0.2 --seed 42
```

说明：

- 支持图片后缀：`jpg`、`jpeg`、`png`、`bmp`、`webp`
- 脚本会自动创建 `images/train`、`images/val`、`labels/train`、`labels/val`
- 划分比例默认为 `8:2`

## 8. 训练命令

```bash
python scripts/train.py --data data/gap_dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --device cpu
```

如果需要使用 GPU：

```bash
python scripts/train.py --data data/gap_dataset/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --device 0
```

训练结果默认保存在：

```text
runs/detect/gap_train/
```

其中常见关键文件包括：

- `weights/best.pt`
- `weights/last.pt`
- `results.png`
- `confusion_matrix.png`

## 9. 验证命令

```bash
python scripts/val.py --weights runs/detect/gap_train/weights/best.pt --data data/gap_dataset/data.yaml
```

验证脚本会输出：

- `Precision`
- `Recall`
- `mAP50`
- `mAP50-95`

## 10. 预测命令

对测试集目录进行识别：

```bash
python scripts/predict.py --weights runs/detect/gap_train/weights/best.pt --source data/gap_dataset/images/test --conf 0.25
```

对单张图片进行识别：

```bash
python scripts/predict.py --weights runs/detect/gap_train/weights/best.pt --source data/gap_dataset/images/test/example.jpg --conf 0.25
```

预测结果默认保存到：

```text
outputs/predict_examples
```

终端还会打印每张图片的识别类别、置信度和检测框坐标，便于在实验报告中记录结果。

## 11. 结果整理命令

```bash
python scripts/export_results.py --run runs/detect/gap_train --pred outputs/predict_examples --out outputs/submit
```

脚本会整理以下内容到 `outputs/submit`：

- `best.pt`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`、`F1_curve.png`、`P_curve.png`、`R_curve.png`（若存在）
- `predict_examples/`
- `README_submit.txt`

## 12. 群内提交建议

建议在群内提交时至少包含以下材料：

1. `best.pt`
2. `results.png`
3. `confusion_matrix.png`
4. 若存在则附上 `PR_curve.png`、`F1_curve.png` 等曲线图
5. 若干张预测示例图
6. 实验报告或实验小结
7. `README_submit.txt`，方便助教快速核对实验内容

如果群内不适合直接发送大模型文件，可以先压缩 `outputs/submit` 后再上传。

## 13. 常见问题排查

### 1. `data.yaml` 存在但仍提示找不到数据

请确认当前工作目录是在 `gap_detection_ultralytics/` 下执行命令。如果你在外层目录运行，请先进入项目目录：

```bash
cd gap_detection_ultralytics
```

### 2. 训练时报模型文件下载失败

可能是网络或版本问题。可以尝试：

- 检查 `ultralytics` 是否安装成功
- 将 `yolo11n.pt` 改为 `yolov8n.pt`
- 提前手动下载模型权重后再通过 `--model` 指定本地路径

### 3. 训练时报数据集格式错误

先运行：

```bash
python scripts/check_dataset.py --data data/gap_dataset/data.yaml
```

根据提示逐项修正标签格式、类别 ID 或坐标范围问题。

### 6.1 代理数据集说明

如果你使用 `scripts/build_proxy_gap_dataset.py` 生成了代理数据集，当前默认会在 `data/gap_dataset` 下生成一套小型样本集。该样本集的默认规模为：

- 训练集：72 张
- 验证集：30 张
- 测试集：30 张

其中训练集和验证集带有三类代理标签，测试集仅保留图片，用于演示预测流程。

当前项目最终保留的数据集就是“类别均衡版严格数据集”，已经放在默认目录 `data/gap_dataset` 下。它的来源是对原始路标检测数据执行严格筛选，并仅在训练集上对少数类进行安全增强后得到。

严格版筛选规则如下：

- `single_gap`：仅保留单目标、位置居中、边界框大小适中的样本
- `multi_gap`：仅保留 2 到 3 个目标且都位于画面中部、边界框大小适中的样本
- `truncated`：仅保留单目标、紧贴图像边界且边界框面积足够大的样本

在最终保留版本中，训练集额外做了类别均衡处理：仅对少数类训练样本执行水平翻转和轻度亮度增强，以缓解 `truncated` 过少的问题。当前这套最终数据规模为：

- 训练集：72 张，其中三类图片数均为 24
- 验证集：79 张
- 测试集：79 张

检查命令：

```bash
python scripts/check_dataset.py --data data/gap_dataset/data.yaml
```

### 4. 预测结果目录里没有图片

请确认：

- `best.pt` 是否存在
- `--source` 指向的文件或目录是否存在
- 测试图片是否是常见格式

### 5. Windows、macOS、Linux 路径兼容性

本项目脚本统一使用 `pathlib.Path` 处理路径，不依赖硬编码反斜杠。命令中的路径参数可按实际系统路径替换。

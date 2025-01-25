

# 基于深度学习的行人重识别

## 项目概述

本项目旨在实现基于深度学习的行人重识别（Person Re-Identification, ReID）功能。行人重识别是计算机视觉领域中的一个重要任务，它的目标是在不同的摄像头视角下，识别出同一个行人。

## 数据集

本项目使用的数据集是Market1501，它包含了1501个不同行人的图像，这些图像是由6个不同的摄像头拍摄的。数据集可以从[这里](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)下载。

## 环境依赖

- Python 3.6+
- PyTorch 1.7+
- torchvision 0.8+
- NumPy
- Matplotlib

## 安装与配置

### 克隆项目

```bash
git clone https://github.com/2221648485/AlignedReID.git
cd AlignedReID
```

### 创建虚拟环境（可选但推荐）

```bash
python -m venv venv
source venv/bin/activate  # 对于Windows用户，使用 `venv\Scripts\activate`
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
AlignedReID/
├── models/
│   └── ResNet.py
├── util/
│   ├── data_manager.py
│   ├── dataset_loader.py
│   └── eval_metrics.py
├── README.md
└── requirements.txt
```

## 主要文件及功能

### 1. `ResNet.py`

包含ResNet模型的定义，主要用于特征提取和分类。关键函数包括：
- `__init__`: 初始化模型，加载预训练的ResNet152模型。
- `forward`: 前向传播函数，根据不同的损失设置返回相应的结果。

### 2. `data_manager.py`

负责数据集的管理和预处理，包括路径检查、数据处理等。关键函数包括：
- `__init__`: 初始化数据集路径，检查路径是否存在，处理训练集、查询集和测试集。
- `check_before_run`: 检查数据集相关目录是否存在。

### 3. `dataset_loader.py`

用于加载数据集，包括读取图像和数据增强等操作。关键函数包括：
- `__init__`: 初始化数据集。
- `__getitem__`: 获取数据集中的一个样本。
- `__len__`: 获取数据集的长度。
- `read_image`: 读取图像文件。

### 4. `eval_metrics.py`

实现了行人重识别的评估指标计算，包括Cuhk03和Market1501两种评估指标。关键函数包括：
- `eval_cuhk03`: 计算Cuhk03评估指标。
- `eval_market1501`: 计算Market1501评估指标。
- `evaluate`: 根据指定的评估指标进行评估。

## 使用方法

### 数据准备

1. 下载Market1501数据集。
2. 将数据集放置在指定的目录下（默认是 `../resource`）。

### 训练模型

```bash
python train_AlignedReID.py
```

## 贡献指南

如果你想为这个项目做出贡献，请遵循以下步骤：
1. Fork这个项目。
2. 创建一个新的分支。
3. 进行你的修改和优化。
4. 提交Pull Request。

## 许可证

本项目使用[许可证名称]许可证，详细信息请查看 `LICENSE` 文件。

## 联系我们

如果你有任何问题或建议，请联系我们。

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
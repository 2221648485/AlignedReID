import torch
from IPython import embed
from torchvision import models
from torch import nn
from torchvision.models import ResNet152_Weights
import torch.nn.functional as F


class ResNet152(nn.Module):
    def __init__(self, num_classes, loss='softmax', **kwargs):
        super().__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1) # 展平
        f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True) + 1e-12) # 归一化
        if not self.training:
            return f
        y = self.classifier(f)
        return y


if __name__ == "__main__":
    model = ResNet152(num_classes=751)
    imgs = torch.Tensor(32, 3, 256, 128)
    f = model(imgs)
    embed()

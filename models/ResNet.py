import torch
from IPython import embed
from torchvision import models
from torch import nn
from torchvision.models import ResNet152_Weights
import torch.nn.functional as F
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d


class ResNet152(nn.Module):
    def __init__(self, num_classes, loss={"softmax"}, aligned=False, **kwargs):
        super().__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.loss = loss
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.model(x)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True) + 1e-12) # 归一化
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


if __name__ == "__main__":
    model = ResNet152(num_classes=751)
    imgs = torch.Tensor(32, 3, 256, 128)
    f = model(imgs)
    print(f.shape)
    embed()

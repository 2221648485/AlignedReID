from torch import nn


class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(x, kernel_size=(1, inp_size[3]))


if __name__ == "__main__":
    import torch
    x = torch.Tensor(32,2048,8,4)
    hmp2d = HorizontalMaxPool2d()
    y = hmp2d(x)
    print(y.shape)


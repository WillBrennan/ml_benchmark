import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, num_in: int, num_out: int, stride: int):
        super().__init__(
            *[
                nn.Conv2d(
                    num_in, num_out, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_out),
                nn.ReLU(inplace=False),
            ]
        )


class DWConv2d(nn.Sequential):
    def __init__(self, num_in: int, num_out: int, stride: int):
        super().__init__(
            *[
                nn.Conv2d(
                    num_in,
                    num_out,
                    3,
                    stride=stride,
                    padding=1,
                    groups=num_out,
                    bias=False,
                ),
                nn.BatchNorm2d(num_out),
                nn.ReLU(inplace=False),
            ]
        )


class DSConv2D(nn.Sequential):
    def __init__(self, num_in: int, num_out: int, stride: int):
        super().__init__(
            *[
                DWConv2d(num_in, num_in, stride),
                nn.Conv2d(num_in, num_out, 1, bias=False),
                nn.BatchNorm2d(num_out),
                nn.ReLU(inplace=False),
            ]
        )

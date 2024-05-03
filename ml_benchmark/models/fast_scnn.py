import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import ConvBNReLU, DSConv2D, DWConv2d


class FastSCNN(nn.Module):
    def __init__(self, weights=None, num_categories: int = 1):
        super().__init__()
        self.learn_to_downsasmple = LearnToDownsample(3)
        self.global_feature_extractor = nn.Sequential(
            BottleNeckModule(64, 64, 2),
            BottleNeckModule(64, 64, 1),
            BottleNeckModule(64, 64, 1),
            BottleNeckModule(64, 96, 2),
            BottleNeckModule(96, 96, 1),
            BottleNeckModule(96, 96, 1),
            BottleNeckModule(96, 128, 1),
            BottleNeckModule(128, 128, 1),
            BottleNeckModule(128, 128, 1),
        )

        self.pooling = PyramidPooling(128)
        self.ffm = FeatureFusion(64, 128, 128, 4)

        self.head = nn.Sequential(
            DSConv2D(128, 128, 1),
            DSConv2D(128, 128, 1),
            nn.Conv2d(128, num_categories, 1),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.learn_to_downsasmple(x)

        x_global = self.global_feature_extractor(x)
        x_global = self.pooling(x_global)

        x = self.ffm(x, x_global)

        x = self.head(x)
        return x


class LearnToDownsample(nn.Sequential):
    def __init__(self, num_in: int):
        super().__init__(
            *[
                ConvBNReLU(num_in, 32, stride=2),
                DSConv2D(32, 48, 2),
                DSConv2D(48, 64, 2),
            ]
        )


class BottleNeckModule(nn.Module):
    def __init__(
        self, num_in: int, num_out: int, stride: int, expansion: int = 6
    ):
        super().__init__()
        num_expand = num_in * expansion
        self.use_shortcut = num_in == num_out and stride == 1

        self.main = nn.Sequential(
            nn.Conv2d(
                num_in, num_expand, 1
            ),  # NOTE(will.brennan): check this is correct!
            DWConv2d(num_expand, num_expand, stride=stride),
            nn.Conv2d(num_expand, num_out, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)

        if self.use_shortcut:
            y = y + x
        return y


class PyramidPooling(nn.Module):
    def __init__(self, num_in: int, sizes: list[int] = [1, 3, 5, 7]):
        super().__init__()
        num_out = num_in // len(sizes)

        def branch(size: int) -> nn.Sequential:
            padding = (size - 1) // 2
            return nn.Sequential(
                *[
                    nn.Conv2d(
                        num_in, num_out, size, padding=padding, bias=False
                    ),
                    nn.BatchNorm2d(num_out),
                    nn.ReLU(inplace=False),
                ]
            )

        self.branches = nn.ModuleList([branch(size) for size in sizes])
        self.project = ConvBNReLU(num_in, num_in, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branched = [branch(x) for branch in self.branches]
        x = torch.cat(branched, dim=1)
        x = self.project(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(
        self,
        num_spatial: int,
        num_global: int,
        num_out: int,
        upsample_factor: int,
    ):
        super().__init__()
        self.upsample_global = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=upsample_factor),
            DWConv2d(num_global, num_global, stride=1),
        )

        self.project_global = nn.Sequential(
            nn.Conv2d(num_global, num_out, kernel_size=1),
            nn.BatchNorm2d(num_out),
        )

        self.project_spatial = nn.Sequential(
            nn.Conv2d(num_spatial, num_out, kernel_size=1),
            nn.BatchNorm2d(num_out),
        )

    def forward(
        self, x_spatial: torch.Tensor, x_global: torch.Tensor
    ) -> torch.Tensor:
        x_global = self.upsample_global(x_global)

        x_global = self.project_global(x_global)
        x_spatial = self.project_spatial(x_spatial)

        x = x_global + x_spatial
        return F.relu(x)

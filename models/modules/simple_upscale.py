import math

import torch
from torch import nn
import torch.nn.functional as F

from models.modules.base import BaseCDModule, BaseCDDecoder


def make_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def upsample_cat(features: list[torch.Tensor]) -> torch.Tensor:
    # upscale all to size of first (largest)
    _, _, h, w = features[0].shape
    feature_map = [features[0]]

    # rescale to dims of first one
    for layer in features[1:]:
        # upscale all to 2x the size of the first (largest)
        resized = F.interpolate(layer, size=(h, w), mode="bilinear")
        feature_map.append(resized)
    # channel-wise concat
    return torch.cat(feature_map, dim=1)


class LinearPixelShuffle(BaseCDDecoder):
    """
    Linear decoder with pixel shuffle to upscale image.
    """

    name = "linPix"

    def __init__(
        self,
        input_sizes: list[int],
        encoder_strides: list[int],
        out_channels: int = 1,
        hidden_size: int = 512,
        layer_n=1,
        input_stages: list[int] = None,
        *args,
        **kwargs,
    ):
        if input_stages is not None:
            self.input_stages = input_stages

            # +1 since stages are 1 indexed
            input_sizes = [
                s for i, s in enumerate(input_sizes) if (i + 1) in input_stages
            ]
            encoder_strides = [
                s for i, s in enumerate(encoder_strides) if (i + 1) in input_stages
            ]
            print(
                f"Limiting decoder input sizes for given stages {input_stages}: {input_sizes}"
            )
            print(
                f"Limiting decoder encoder strides for given stages {input_stages}: {encoder_strides}"
            )

        super().__init__(
            input_sizes=input_sizes,
            encoder_strides=encoder_strides,
            out_channels=out_channels,
            *args,
            **kwargs,
        )

        # concat if more than 1
        input_channels = sum(self.input_sizes)
        hidden_channels = hidden_size
        # match first scale
        actual_stride = int(self.encoder_strides[0])

        layers = []
        for i in range(layer_n - 1):
            layers.append(make_block(input_channels, hidden_channels, kernel_size=3))
            input_channels = hidden_channels

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=int(actual_stride**2 * out_channels),
                    kernel_size=1,
                    padding="same",
                ),
                nn.PixelShuffle(actual_stride),
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        features = list(x.values())
        feat_map = upsample_cat(features)
        return self.layers(feat_map)

    def get_parameters(self, pretraining=False):
        return [{"params": self.layers.parameters()}]


class CatUpsampConv(BaseCDDecoder):
    """
    Simple net that concats features from all layers and upsamples using bilinear upsampling and conv.

    """

    name = "catUp"

    def __init__(
        self,
        input_sizes: list[int],
        out_channels: int,
        out_size: int,
        encoder_strides: list[int],
        layer_n=1,
        scale_channels=True,
    ):
        super().__init__()
        self.layers = nn.Sequential()

        curr_channels = sum(input_sizes)

        for i in range(layer_n - 1):
            # this gives upscaling factor needed to get to original input in about equal steps.
            # "about" - since scaling with different encoder strides requires different layer_n for "correct" scale
            scale_factor = math.ceil(encoder_strides[-1] ** (1 / layer_n))

            if scale_channels:
                next_channels = int(curr_channels / scale_factor)
            else:
                next_channels = curr_channels

            # reduce channels up the chain
            self.layers.append(
                self.make_upscale_block(
                    curr_channels, next_channels, factor=scale_factor
                )
            )
            curr_channels = next_channels

        self.layers.append(
            nn.Sequential(
                nn.Upsample(size=out_size, mode="bilinear"),
                nn.Conv2d(
                    in_channels=curr_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                ),
            )
        )

    @staticmethod
    def make_upscale_block(in_channels, out_channels, factor):
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="bilinear"),
            make_block(in_channels, out_channels, kernel_size=3),
        )

    def forward(self, features):
        features = list(features.values())
        feature_map = upsample_cat(features)

        return self.layers(feature_map)

    def get_parameters(self, pretraining=False):
        return [{"params": self.layers.parameters()}]


class SimpleConv(BaseCDDecoder):
    """
    Simple net that concats features from all layers and applies conv, followed by upsampling.

    """

    name = "simpConv"

    def __init__(
        self,
        input_sizes: list[int],
        out_channels: int,
        out_size: int,
        encoder_strides: list[int],
        layer_n=1,
        scale_channels=True,
    ):
        super().__init__()
        self.layers = nn.Sequential()

        curr_channels = sum(input_sizes)

        for i in range(layer_n - 1):
            # this gives upscaling factor needed to get to original input in about equal steps.
            # "about" - since scaling with different encoder strides requires different layer_n for "correct" scale
            scale_factor = math.ceil(encoder_strides[-1] ** (1 / layer_n))

            if scale_channels:
                next_channels = int(curr_channels / scale_factor)
            else:
                next_channels = curr_channels

            # reduce channels up the chain
            self.layers.append(
                make_block(
                    in_channels=curr_channels, out_channels=next_channels, kernel_size=3
                )
            )
            curr_channels = next_channels

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=curr_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                ),
                nn.Upsample(size=out_size, mode="bilinear"),
            )
        )

    def forward(self, features):
        features = list(features.values())
        feature_map = upsample_cat(features)

        return self.layers(feature_map)

    def get_parameters(self, pretraining=False):
        return [{"params": self.layers.parameters()}]

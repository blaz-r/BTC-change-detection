"""Adapted from MTP.
https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/opencd/models/decode_heads/unet_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import BaseCDDecoder


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm and use_batchnorm != "inplace":
            # bn = nn.BatchNorm2d(out_channels)
            # _, bn = build_norm_layer(norm_cfg, out_channels)
            bn = nn.SyncBatchNorm(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, norm_cfg=None):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            norm_cfg,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UNetHead(BaseCDDecoder):
    """
    Adapted from MTP OpenCD config.
    https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/opencd/models/decode_heads/unet_head.py
    https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/configs/mtp/oscd/rvsa-b-unet-96-mae-mtp_oscd_rgb.py
    """

    name = "OCD_Unet"

    def __init__(
        self,
        input_sizes,
        encoder_strides,
        out_channels: int,  # num_classes in MTP
        decoder_channels=[512, 256, 128, 64],
        n_blocks=4,
        channels=64,
        dropout_ratio=0.1,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        align_corners=False,  # unused since we don't do concat
        in_channels=[768, 768, 768, 768],  # unused in MTP
        in_index=[0, 1, 2, 3],  # unused in MTP
        *args,
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes,
            encoder_strides=encoder_strides,
            out_channels=out_channels,
            *args,
            **kwargs,
        )

        self.n_blocks = n_blocks
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[1:] # (64, 256, 512, 1024, 2048); (64, 128, 256, 512)
        # reverse channels to start from head of encoder
        input_sizes = input_sizes[
            ::-1
        ]  # (2048, 1024, 512, 256, 64); (512, 256, 128 ,64)

        # computing blocks input and output channels
        head_channels = input_sizes[0]  # 2048; 512
        in_channels = [head_channels] + list(
            decoder_channels[:-1]
        )  # [2048, 256, 128, 64, 32]; [512, 256, 128, 64]
        skip_channels = list(input_sizes[1:]) + [
            0
        ]  # [1024, 512, 256, 64, 0]; [256, 128, 64, 0]
        out_channels = decoder_channels  # (256, 128, 64, 32, 16) # (256, 128, 64, 32)

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.01)  # same as in MTP
            if module.bias is not None:
                module.bias.data.zero_()

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, features):
        features = list(features.values())
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        if self.encoder_strides[-1] != 2**self.n_blocks:
            # since each block upsamples by 2, final upscal is 2**nB,
            # if that doesn't match final encoder stride (swin), do one more
            x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x = self.cls_seg(x)

        return x

    def get_parameters(self, pretraining) -> list[dict]:
        return [{"params": self.parameters(), **self.get_lr_dict()}]


if __name__ == "__main__":
    unet = UNetHead(
        input_sizes=[96, 96 * 2, 96 * 4, 96 * 8],
        decoder_channels=[512, 256, 128, 64],
        encoder_strides=[2, 2, 2, 2],
        pretraining=False,
    )

    feat = [
        torch.rand(8, 96, 64, 64),
        torch.rand(8, 96 * 2, 32, 32),
        torch.rand(8, 96 * 4, 16, 16),
        torch.rand(8, 96 * 8, 8, 8),
    ]

    out = unet(feat)
    print(out.shape)

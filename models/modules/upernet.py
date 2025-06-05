# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch UperNet model. From huggingface https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/modeling_upernet.py#L339."""

from typing import List, Tuple, Union

import torch
from torch import nn
from transformers import UperNetForSemanticSegmentation

from models.modules import BaseCDDecoder


class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales: Tuple[int, ...],
        in_channels: int,
        channels: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(
                pool_scale=pool_scale, in_channels=in_channels, channels=channels
            )
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetFCNHead(nn.Module):
    """
    Hugging facee Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=512,
        num_convs=1,
        in_index: int = 2,
        concat=False,
        out_channels=1,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = hidden_channels
        self.num_convs = num_convs
        self.concat_input = concat
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        self.classifier = nn.Conv2d(self.channels, out_channels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


class UperNetHead(BaseCDDecoder):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    name = "upernet"

    def __init__(
        self,
        input_sizes,
        encoder_strides,
        out_channels=1,
        hidden_size=512,
        out_size=None,
        use_auxfcn=False,
        load_weights: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes,
            encoder_strides=encoder_strides,
            out_channels=out_channels,
            *args,
            **kwargs,
        )

        self.pool_scales = (1, 2, 3, 6)  # e.g. (1, 2, 3, 6)
        self.in_channels = [int(c) for c in input_sizes]
        self.channels = hidden_size
        self.align_corners = False
        self.out_size = out_size
        self.classifier = nn.Conv2d(self.channels, out_channels, kernel_size=1)

        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        # psp bottleneck
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UperNetConvModule(
                self.channels, self.channels, kernel_size=3, padding=1
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        if use_auxfcn:
            # take index2 or if there is less inputs, take index of last
            fcn_idx = min(2, len(input_sizes) - 1)
            self.fcn = UperNetFCNHead(
                in_channels=input_sizes[fcn_idx], out_channels=out_channels
            )
            self.fcn.init_weights()
            if fcn_idx < 2:
                print(f"Using index {fcn_idx} in UperNet auxiliary FCN head!!!")
            self.name += "-FCNaux"
        else:
            self.fcn = None

        if load_weights is None:
            self.init_weights()
        else:
            self.load_weights(load_weights)
            self.name += "PT"

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def load_weights(self, load_weights):
        if load_weights.startswith("HF"):
            name = load_weights.split(":")[1]
            head = UperNetForSemanticSegmentation.from_pretrained(name).decode_head
            uper_state = head.state_dict()
        else:
            all_states = torch.load(load_weights)["state_dict"]
            # remove backbone things and keep only upernet names
            uper_state = {
                ".".join(f.split(".")[1:]): w
                for f, w in all_states.items()
                if f.startswith("decode_head")
            }
        # change out seg, since this dims don't match
        uper_state = {
            f: w for f, w in uper_state.items() if not f.startswith("classifier")
        }

        missing, unexpected = self.load_state_dict(uper_state, strict=False)
        print(f"Loaded weights from {load_weights}")
        print(f"Missing: {missing}")
        print(f"Unexpected: {unexpected}")

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(
        self, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        encoder_hidden_states = list(encoder_hidden_states.values())

        # build laterals - conv feat
        laterals = [
            lateral_conv(encoder_hidden_states[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top layer is PPMed
        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path, take conved (lateral) from prev stage and uspcale
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        # to same scale as largest
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        output = nn.functional.interpolate(
            output, self.out_size, mode="bilinear", align_corners=self.align_corners
        )

        if self.fcn is not None and self.training:
            fcn_out = self.fcn(encoder_hidden_states)
            fcn_out = nn.functional.interpolate(
                fcn_out,
                self.out_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            return output, fcn_out

        return output

    def get_parameters(self, pretraining) -> list[dict]:
        return [{"params": self.parameters(), **self.get_lr_dict()}]

    def transfer_from_pretrained(self, state_dict):
        # remove these due shape mismatch
        state_dict.pop("classifier.weight")
        state_dict.pop("classifier.bias")

        super().transfer_from_pretrained(state_dict)

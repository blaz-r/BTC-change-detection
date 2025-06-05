import importlib

import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as T
import torch

import albumentations as A


def build_transforms(config, pretrain: bool, test: bool, has_mask: bool = True):
    return DualTransform(config, pretrain, test, has_mask)


class DualTransform:
    def __init__(self, config, pretrain: bool, test: bool, has_mask: bool):
        self.img_size = (config.data.img_size, config.data.img_size)
        self.pretrain = pretrain
        self.test = test
        self.has_mask = has_mask
        # always resize
        transforms = [A.Resize(*self.img_size)]

        # build other transforms instances from config
        transforms.extend(self.build_from_config(config))

        assert isinstance(transforms[-1], A.Normalize), (
            "Normalize must be included in the transforms and on last position"
        )
        transforms.append(ToTensorV2())

        assert not test or len(transforms) == 3, (
            "Testing transforms should only contain resize, normalize and ToTensor"
        )

        proc_str = "testing" if test else "training"
        print(f"Transforms used for {proc_str}:")
        for tr in transforms:
            print(tr)

        if self.has_mask:
            add = {"imageB": "image", "label": "mask"}
        else:
            add = {"imageB": "image"}

        self.transforms = A.Compose(
            transforms,
            additional_targets=add,
        )

    def build_from_config(self, config) -> list:
        transforms = []
        if self.pretrain:
            transform_config = config.pretrain.get("transforms", [])
            # emtpy list or not present
            if not transform_config:
                print("No pretrain transforms, using training transforms")
                transform_config = config.train.transforms
        else:
            transform_config = config.train.transforms

        if self.test:
            # only take last transform which is normalisation
            tr_config_list = [transform_config[-1]]
        else:
            tr_config_list = transform_config

        for tr_conf in tr_config_list:
            cls_name, init_args = next(iter(tr_conf.items()))
            if "size" in init_args:
                init_args["size"] = self.img_size

            tr_cls = getattr(importlib.import_module("albumentations"), cls_name)
            transforms.append(tr_cls(**init_args))

        return transforms

    def __call__(self, data):
        if self.has_mask:
            transformed = self.transforms(
                image=data["imageA"], imageB=data["imageB"], label=data["label"]
            )
        else:
            transformed = self.transforms(image=data["imageA"], imageB=data["imageB"])

        # transformed["imageA_unnorm"] = data["imageA"]
        # transformed["imageB_unnorm"] = data["imageB"]
        # transformed["img_idx"] = data["img_idx"]
        # # get to 0-1 range and add channel dim
        # transformed["label"] = (transformed["label"] / 255).unsqueeze(0)

        transformed_batch = {
            "imageA": transformed["image"],
            "imageB": transformed["imageB"],
            # "imageA_unnorm": data["imageA"],
            # "imageB_unnorm": data["imageB"],
            # "img_idx": data["img_idx"],
        }
        if self.has_mask:
            transformed_batch["label"] = (transformed["label"] / 255).unsqueeze(0)
        # for iA, iB, lbl in zip(batch["imageA"], batch["imageB"], batch["label"]):
        #
        #
        #     transformed_batch["imageA"].append(transformed["image"])
        #     transformed_batch["imageB"].append(transformed["imageB"])
        #     # get to 0-1 range and add channel dim
        #     transformed_batch["label"].append((transformed["label"] / 255).unsqueeze(0))

        return transformed_batch

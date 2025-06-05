import copy

from models.modules.base import BaseCDDecoder, BaseCDDiff
from models.modules.hf_backbone import HFBackbone, build_hfbackbone
from models.modules.other_pt import OtherPT, build_other_pt

import importlib


def build_encoder(config, logger, pretraining):
    model_type = config.encoder.class_path.split(".")[-1]
    if model_type == HFBackbone.__name__:
        return build_hfbackbone(config, pretraining=pretraining)
    elif model_type == OtherPT.__name__:
        return build_other_pt(config)
    else:
        raise ValueError(f"Model type {model_type} not supported")


def build_diff(config, dims, resolutions, pretraining):
    split = config.diff.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    diff: BaseCDDiff = getattr(importlib.import_module(module_path), class_name)
    diff_args = config.diff.get("init_args", {})
    return diff(
        dims=dims, resolutions=resolutions, pretraining=pretraining, **diff_args
    )


def build_pre_diff(config, dims, resolutions, pretraining):
    if config.pre_diff is None:
        return None
    split = config.pre_diff.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    diff: BaseCDDiff = getattr(importlib.import_module(module_path), class_name)
    diff_args = config.pre_diff.get("init_args", {})
    return diff(
        dims=dims, resolutions=resolutions, pretraining=pretraining, **diff_args
    )


def build_decoder(config, input_sizes, encoder_strides, pretraining):
    config = copy.deepcopy(config)
    if pretraining:
        if "pretrain" in config.decoder:
            subconf = config.decoder.pretrain
        else:
            print("!! Using same decoder for pretraining as in training.")
            subconf = config.decoder.train
    else:
        subconf = config.decoder.train

    split = subconf.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    decoder = getattr(importlib.import_module(module_path), class_name)

    return decoder(
        out_size=config.data.img_size,
        input_sizes=input_sizes,
        encoder_strides=encoder_strides,
        pretraining=pretraining,
        **subconf.init_args.as_dict(),
    )


def build_out_proc(config, pretraining):
    if config.out_proc is None:
        return None
    split = config.out_proc.class_path.split(".")
    module_path, class_name = ".".join(split[:-1]), split[-1]

    out_proc = getattr(importlib.import_module(module_path), class_name)
    out_proc_args = config.out_proc.get("init_args", {})
    return out_proc(pretraining=pretraining, **out_proc_args)

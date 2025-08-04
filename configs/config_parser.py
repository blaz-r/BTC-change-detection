from attr import dataclass
from jsonargparse import ArgumentParser, ActionConfigFile

from models.modules.base import (
    BaseCDEncoder,
    BaseCDDiff,
    BaseCDDecoder,
    BaseCDOutProc,
    BaseCDPreDiff,
)


def get_parser():
    parser = ArgumentParser("Change oriented remote sensing")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )

    add_arguments(parser)
    add_framework_arguments(parser)

    return parser


@dataclass
class SchedulerArgs:
    type: str
    gamma: float = None
    ratio_t0: float = None
    warmup_epoch_ratio: float = None
    warmup_lr_ratio: float = None


@dataclass
class TrainArgs:
    epochs: int
    val_freq: int
    base_lr: float
    weight_decay: float
    layer_decay: float
    lr_scheduler: SchedulerArgs
    loss: list[str]
    transforms: list[dict]
    ckpt: str | None = None
    grad_clip_val: float = None


@dataclass
class DataArgs:
    dataset: str
    img_size: int
    num_workers: int
    batch_size: int
    pin_memory: bool
    data_path: str = "./datasets"
    use_hf: bool = True
    load_in_mem: str | None = None


def add_arguments(parser: ArgumentParser):
    """
    Add cli only args, used for various small tweaks.

    Args:
        parser: parser object
    """
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--devices", default="auto", help="devices")
    parser.add_argument("--tag", type=str, default="", help="tag of experiment")
    parser.add_argument("--config_tag", type=str, help="tag of config file used")
    parser.add_argument(
        "--res_path", type=str, default="./results", help="tag of config file used"
    )
    parser.add_argument("--dev", help="dev run", action="store_true")
    parser.add_argument(
        "--wandb_proj",
        type=str,
        default=None,
        help="wandb project name, will override default set in code",
    )

    parser.add_argument(
        "--vis_path",
        default=False,
        type=bool,
        help="Flag whether images are visualised (True / False).",
    )
    parser.add_argument("--ckpt_path", help="save checkpoints", type=str, default=None)
    parser.add_argument(
        "--eval_only",
        help="run only evaluation. Note: requires mandatory weight passed with --ckpt_path",
        action="store_true",
    )


def add_framework_arguments(parser: ArgumentParser):
    parser.add_subclass_arguments(
        BaseCDEncoder,
        "encoder",
        required=True,
        skip={"config", "logger", "pretraining"},
    )
    parser.add_subclass_arguments(
        BaseCDDiff, "diff", required=True, skip={"dims", "resolutions", "pretraining"}
    )
    parser.add_subclass_arguments(
        BaseCDPreDiff,
        "pre_diff",
        required=False,
        skip={"dims", "resolutions", "pretraining"},
    )
    parser.add_subclass_arguments(
        BaseCDDecoder,
        "decoder.train",
        required=True,
        skip={"out_size", "input_sizes", "encoder_strides", "pretraining"},
    )
    parser.add_subclass_arguments(
        BaseCDOutProc, "out_proc", required=False, skip={"pretraining"}
    )
    parser.add_argument("--data", type=DataArgs)
    parser.add_argument("--train", type=TrainArgs)

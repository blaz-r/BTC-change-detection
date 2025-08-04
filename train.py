from pathlib import Path

import pandas as pd
import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryJaccardIndex,
)

from configs.config_parser import get_parser
from data.datamodule import CDDataModule
from models.callbacks.visualiser import Visualizer

from models.finetune_framework import FinetuneFramework


def finetune(framework, datamodule, config, wandb_logger):
    has_cuda = torch.cuda.is_available()

    if config.ckpt_path != "None":
        ckpt_callback = ModelCheckpoint(
            dirpath=f"{config.ckpt_path}/{config.seed}/{config.config_tag}_{config.tag}_{config.data.dataset}",
            filename="weights",
            save_weights_only=True,
        )
        callbacks = [ckpt_callback]
    else:
        callbacks = []

    trainer = Trainer(
        max_epochs=config.train.epochs,
        check_val_every_n_epoch=config.train.val_freq,
        logger=wandb_logger,
        accelerator="auto",
        devices=config.devices if has_cuda else "auto",
        callbacks=callbacks,
        precision="16-mixed" if has_cuda else None,
        fast_dev_run=config.dev,
        gradient_clip_val=config.train.grad_clip_val,
        gradient_clip_algorithm="norm",
        # strategy='ddp_find_unused_parameters_true'
    )

    trainer.fit(
        model=framework,
        datamodule=datamodule,
    )


def load_weights(config, finetune_framework, res_path):
    if config.ckpt_path.startswith("blaz-r/"):
        print(
            f"Loading checkpoint from huggingface {config.ckpt_path}. Make sure that the config matches the checkpoint!!"
        )
        if config.data.dataset not in config.ckpt_path:
            msg = f"Checkpoint dataset and config dataset missmatch. Expected {config.data.dataset}, but got {config.ckpt_path}"
            raise ValueError(msg)
        finetune_framework = finetune_framework.from_pretrained(
            config.ckpt_path,
            metrics=MetricCollection(
                {
                    "F1": BinaryF1Score(),
                    "Recall": BinaryRecall(),
                    "Precision": BinaryPrecision(),
                    "cIoU": BinaryJaccardIndex(),
                }
            ),
            logger=None,
            config_namespace=config,  # not the cleanest thing, but this way we keep the passed config settings.
        )
        # if loading from HF, the class is re-instantiated so we need to set this
        finetune_framework.res_path = res_path
    else:
        print(
            f"Loading checkpoint from file system: {config.ckpt_path}. Make sure that the config matches the checkpoint!!"
        )
        state_dict = torch.load(config.ckpt_path, weights_only=False)["state_dict"]
        finetune_framework.load_state_dict(state_dict)

    return finetune_framework


def evaluate(framework, datamodule, config, wandb_logger):
    callbacks = None
    if config.vis_path:
        if config.seed == 42:
            visualizer = Visualizer(
                res_path=framework.res_path,
                criterion=BinaryF1Score(
                    zero_division=1
                ),  # criterion for plotting, if f1 < 1
                criterion_limit=1,
            )
            callbacks = [visualizer]
        else:
            print(f"Skipping visualisation since seed {config.seed} != 42")

    test_trainer = Trainer(
        logger=None if config.dev else wandb_logger,
        accelerator="auto",
        devices=[0] if torch.cuda.is_available() else "auto",
        fast_dev_run=config.dev,
        callbacks=callbacks,
    )
    results = test_trainer.test(model=framework, datamodule=datamodule)
    pd.DataFrame(results).to_csv(framework.res_path / "res.csv", index=False)


def main(seed=None):
    parser = get_parser()
    config = parser.parse_args()
    if seed is not None:
        config.seed = seed

    seed_everything(config.seed)

    datamodule = CDDataModule(
        config,
        data_path=config.data.data_path,
        pretrain=False,
        use_hf=config.data.use_hf,
        load_in_mem=config.data.load_in_mem,
    )

    finetune_framework = FinetuneFramework(
        config_namespace=config,
        config=config.as_dict(),
        metrics=MetricCollection(
            {
                "F1": BinaryF1Score(),
                "Recall": BinaryRecall(),
                "Precision": BinaryPrecision(),
                "cIoU": BinaryJaccardIndex(),
            }
        ),
        logger=None,
    )

    res_path = finetune_framework.generate_and_verify_res_path(
        Path(config.res_path) / str(config.seed)
    )
    finetune_framework.dump_config()

    run_name = finetune_framework.generate_exp_name()
    if config.wandb_proj != "None":
        wandb_logger = WandbLogger(
            project=config.wandb_proj, log_model=False, name=run_name
        )
    else:
        wandb_logger = None

    if config.eval_only:
        finetune_framework = load_weights(config, finetune_framework, res_path)
    else:
        print(f"Starting finetune for {run_name}")
        finetune(finetune_framework, datamodule, config, wandb_logger)

    print(f"Starting eval for {run_name}")
    evaluate(finetune_framework, datamodule, config, wandb_logger)

    wandb.finish()


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    main()

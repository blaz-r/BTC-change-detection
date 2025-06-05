from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torchvision.ops.focal_loss import sigmoid_focal_loss

from models.framework import Framework
from models.loss.dice import dice_loss_smooth


class FinetuneFramework(Framework):
    pretraining = False

    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def generate_and_verify_res_path(self, save_path):
        if self.res_path is None:
            config = self.config
            # omit dataset name and config tag from name, rather have them in path
            res_run_name = f"{config.tag}_[{self.enc.name}_{self.diff.name}_{self.dec.name}_{self.loss_name}]"

            if self.pre_diff is not None:
                res_run_name += f"({self.pre_diff.name})"
            if self.out_proc is not None:
                res_run_name += f"({self.out_proc.name})"

            res_path = (
                Path(save_path) / config.config_tag / res_run_name / config.data.dataset
            )
            # raise exception if exists in non dev mode
            res_path.mkdir(parents=True, exist_ok=config.dev)

            self.res_path = res_path

        return self.res_path

    def dump_config(self):
        with (self.res_path / "ft_config.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

    def build_loss(self):
        # build dict of losses so we can calculate individual named terms when using composite losses (e.g. ce + focal)
        loss_list = self.config.train.loss

        loss_fn_dict = {}
        name = ""

        for i, l in enumerate(loss_list):
            name += l
            if i < len(loss_list) - 1:
                name += "-"

            if l == "focal":
                loss_fn_dict[l] = partial(
                    sigmoid_focal_loss, reduction="mean", alpha=-1
                )
            elif l == "ce":
                loss_fn_dict[l] = torch.nn.BCEWithLogitsLoss()
            elif l == "2chce":
                f = torch.nn.CrossEntropyLoss()

                def two_ch_ce(pred, t):
                    return f(pred, t.squeeze().long())

                loss_fn_dict[l] = two_ch_ce
            elif l == "dice":
                loss_fn_dict[l] = dice_loss_smooth
            else:
                raise ValueError(f"Unknown loss function {l}")

        def loss_callable(preds, target):
            total_loss = 0
            loss_terms = {}
            for name, curr_loss_f in loss_fn_dict.items():
                curr_loss_val = curr_loss_f(preds, target)
                loss_terms[name] = curr_loss_val.item()
                total_loss += curr_loss_val
            return total_loss, loss_terms

        return loss_callable, name

    def forward(self, batch):
        i1, i2 = batch["imageA"], batch["imageB"]
        x = torch.cat([i1, i2], dim=0)

        if self.in_proc:
            x = self.in_proc(x)
        x = self.enc(x)
        if self.pre_diff:
            x = self.pre_diff(x)
        if self.diff:
            x = self.diff(x)
        x = self.dec(x)
        if self.out_proc:
            x = self.out_proc(x)

        return x

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        lbl = batch["label"]

        x = self.forward(batch)

        if isinstance(x, tuple):
            # also apply to aux head
            aux_loss, aux_loss_terms = self.criterion(x[0], lbl)
            loss, loss_terms = self.criterion(x[1], lbl)
            # add total
            loss = loss + aux_loss
            # include aux in dict
            for n, v in aux_loss_terms.items():
                loss_terms[f"aux_{n}"] = v
        else:
            loss, loss_terms = self.criterion(x, lbl)

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log_dict(loss_terms, prog_bar=False, logger=True)

        return {"loss": loss}

    def predict_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        x = self.forward(batch)

        if len(x.shape) == 4 and x.shape[1] == 2:
            # 2 channel predictions (unused in our experiments)
            pred = torch.softmax(x, dim=1)[:, 1].unsqueeze(1)
            return pred
        else:
            return torch.sigmoid(x)

    def test_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        batch["pred"] = self.predict_step(batch, *args, **kwargs)

        self.metrics.update(batch["pred"], torch.where(batch["label"] > 0.5, 1, 0))
        self.log_dict(self.metrics, on_epoch=True, prog_bar=True)

        return batch

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        try:
            results = self.test_step(batch, *args, **kwargs)
        except Exception as e:
            # catch only in val, test should always pass
            print(
                f"Error in validation {e}, validation metrics will now be unreliable, rely on test metrics."
            )
            results = None

        return results

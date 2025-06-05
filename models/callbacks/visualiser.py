from pathlib import Path
from typing import Any

import cv2
import torch
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from torchmetrics import Metric


class Visualizer(Callback):
    def __init__(
        self,
        res_path: Path,
        criterion: Metric | None = None,
        criterion_limit: float = 0.5,
    ):
        self.save_path = res_path / "visual"
        self.save_path.mkdir(exist_ok=False)

        self.criterion = criterion
        self.criterion_limit = criterion_limit

    def visualize(self, predictions, batch):
        for idx, pred_map, gt_mask, imageA, imageB in zip(
            batch["img_idx"],
            predictions.detach().cpu(),
            batch["label"].detach().cpu(),
            batch["imageA_unnorm"].cpu().numpy(),
            batch["imageB_unnorm"].cpu().numpy(),
        ):
            plot_current = True
            val = None
            if self.criterion is not None:
                val = self.criterion(pred_map, torch.where(gt_mask > 0.5, 1, 0))

                plot_current = val < self.criterion_limit

            if plot_current:
                # plot only if criterion indicates a poor sample
                fig, plots = plt.subplots(2, 3, figsize=(9, 6))
                for s_plt in plots.flatten():
                    s_plt.axis("off")

                fig.tight_layout()

                plots[0][0].imshow(imageA)
                plots[0][0].title.set_text("Pre-Image")

                plots[1][0].imshow(imageB)
                plots[1][0].title.set_text("Post-Image")

                plots[0][1].imshow(gt_mask.squeeze().numpy())
                plots[0][1].title.set_text("GT mask")

                plots[1][1].imshow(
                    (pred_map.squeeze().numpy() > 0.5), vmax=1, vmin=0, cmap="gray"
                )
                plots[1][1].title.set_text("Pred. mask")

                if val is not None:
                    plots[0][2].title.set_text(f"F1: {val:.3f}")

                plots[1][2].imshow(pred_map.squeeze().numpy(), vmax=1, vmin=0)
                plots[1][2].title.set_text("Pred. map")

                fig.tight_layout()

                plt.savefig(self.save_path / f"{idx}.jpg", bbox_inches="tight")

            pred_maps_dir = self.save_path / "pred_map"
            pred_maps_dir.mkdir(exist_ok=True, parents=True)

            # save as png for lossless mask
            cv2.imwrite(
                str(pred_maps_dir / f"{idx}.png"), pred_map.squeeze().numpy() * 255
            )

            plt.close("all")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.visualize(predictions=outputs["pred"], batch=batch)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)

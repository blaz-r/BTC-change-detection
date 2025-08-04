import lightning as L
from huggingface_hub import PyTorchModelHubMixin
from argparse import Namespace
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LinearLR,
    PolynomialLR,
    ExponentialLR,
)
from torchmetrics import MetricCollection

from models.modules import (
    build_encoder,
    build_diff,
    build_decoder,
    build_out_proc,
    build_pre_diff,
)


class Framework(
    L.LightningModule,
    PyTorchModelHubMixin,
    repo_url="https://github.com/blaz-r/BTC-change-detection",
    paper_url="https://arxiv.org/abs/2507.03367",
    docs_url="https://github.com/blaz-r/BTC-change-detection",
):
    pretraining = None  # setup in subclass

    def __init__(
        self,
        config_namespace: Namespace,
        config: dict,  # to save in wandb
        logger,
    ):
        super().__init__()
        self.config = config_namespace
        self.config_dict = config  # to save in wandb
        self.metrics: MetricCollection

        # make dec and get dims & strides
        self.enc = build_encoder(config_namespace, logger, pretraining=self.pretraining)
        dims = self.enc.get_out_dims()  # channel dims
        strides = self.enc.get_strides()  # factor of reduction from original image
        resolutions = [int(self.config.data.img_size / s) for s in strides]
        # get diff
        self.diff = build_diff(
            config_namespace,
            dims=dims,
            resolutions=resolutions,
            pretraining=self.pretraining,
        )
        # adjust channel dims (since some diffs like catdiff expand channel dims)
        dims = self.diff.adjust_dims(dims)
        # prepare decoder
        self.dec = build_decoder(
            config_namespace,
            input_sizes=dims,
            encoder_strides=strides,
            pretraining=self.pretraining,
        )

        self.in_proc = None  # unused
        # optional, in our exps unused
        self.pre_diff = build_pre_diff(
            config_namespace,
            dims=dims,
            resolutions=resolutions,
            pretraining=self.pretraining,
        )
        # optional, in our exps unused
        self.out_proc = build_out_proc(config_namespace, pretraining=self.pretraining)

        # build loss
        self.criterion, self.loss_name = self.build_loss()

        self.res_path = None
        self.exp_name = None

        self.save_hyperparameters(ignore=["metrics", "module_list", "config_namespace"])

    def generate_exp_name(self):
        """Return experiment name with and without dataset and configTag for visuals."""
        config = self.config
        self.exp_name = (
            f"{config.config_tag}_{config.tag}_{config.data.dataset}"
            + f"_[{self.enc.name}_{self.diff.name}_{self.dec.name}_{self.loss_name}]"
        )
        if self.pre_diff is not None:
            self.exp_name += f"({self.pre_diff.name})"
        if self.out_proc is not None:
            self.exp_name += f"({self.out_proc.name})"
        return self.exp_name

    def build_loss(self):
        # implement in subclass
        raise NotImplementedError

    def configure_optimizers(self):
        params = []
        missing = []
        for name in [
            "in_proc",
            "enc",
            "pre_diff",
            "diff",
            "dec",
            "out_proc",
            "strategy",
        ]:
            module = getattr(self, name, None)
            if module:
                params += module.get_parameters(pretraining=self.pretraining)
            else:
                missing.append(name)

        print(f"Following modules not used: {missing}")

        if self.pretraining:
            # unused
            config = self.config.pretrain
        else:
            config = self.config.train

        optimizer = AdamW(
            params,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        lr_scheduler = self.get_scheduler(optimizer, config)

        if lr_scheduler is None:
            return optimizer

        return [optimizer], [lr_scheduler]

    def get_scheduler(self, optimizer, config):
        if config.lr_scheduler.type == "none":
            print("Lr scheduler not used")
            return None
        elif config.lr_scheduler.type == "multistep":
            assert config.lr_scheduler.gamma
            steps = [int(0.8 * config.epochs), int(0.9 * config.epochs)]
            print(f"Using multistep LR with steps{steps}")
            return MultiStepLR(
                optimizer,
                milestones=steps,
                gamma=config.lr_scheduler.gamma,
            )
        elif config.lr_scheduler.type == "cosine":
            assert config.lr_scheduler.ratio_t0
            # if ratio_t0 = 1 -> cosine annealing no restart
            t0 = int(config.lr_scheduler.ratio_t0 * config.epochs)
            print(f"Using cosine scheduler with T0={t0}")
            factor = 1e-3
            min_lr = config.base_lr * factor
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                eta_min=min_lr,
            )
        elif config.lr_scheduler.type == "linear":
            factor = 1e-3
            # from base to base * 0.1e-3
            return LinearLR(
                optimizer, start_factor=1, end_factor=factor, total_iters=config.epochs
            )
        elif config.lr_scheduler.type == "poly":
            assert config.lr_scheduler.gamma  # power in this case
            return PolynomialLR(
                optimizer, power=config.lr_scheduler.gamma, total_iters=config.epochs
            )
        elif config.lr_scheduler.type == "exp":
            assert config.lr_scheduler.gamma
            return ExponentialLR(optimizer, gamma=config.lr_scheduler.gamma)
        elif config.lr_scheduler.type == "cosine_warmup":
            assert config.lr_scheduler.ratio_t0
            assert config.lr_scheduler.warmup_epoch_ratio
            assert config.lr_scheduler.warmup_lr_ratio
            factor = config.lr_scheduler.warmup_lr_ratio
            min_lr = config.base_lr * factor
            warmup_epochs = int(config.lr_scheduler.warmup_epoch_ratio * config.epochs)

            # if ratio_t0 = 1 -> cosine annealing no restart
            t0 = int(config.lr_scheduler.ratio_t0 * config.epochs)
            print(
                f"Using cosine scheduler with T0={t0} and {warmup_epochs} epoch warmup"
            )

            # adjust for warmup:
            t0 -= warmup_epochs

            # start from lr * factor and move towards base lr for 'warmup_ratio * epoch' epochs
            warmup = LinearLR(
                optimizer, start_factor=factor, end_factor=1, total_iters=warmup_epochs
            )
            cosine_lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer, [warmup, cosine_lr_scheduler], milestones=[warmup_epochs]
            )
        elif config.lr_scheduler.type == "multistep_warmup":
            assert config.lr_scheduler.gamma
            assert config.lr_scheduler.warmup_epoch_ratio
            assert config.lr_scheduler.warmup_lr_ratio
            steps = [int(0.8 * config.epochs), int(0.9 * config.epochs)]
            factor = config.lr_scheduler.warmup_lr_ratio
            warmup_epochs = int(config.lr_scheduler.warmup_epoch_ratio * config.epochs)
            print(
                f"Using multistep LR with steps {steps} and {warmup_epochs} epoch warmup"
            )

            # adjust for warmup
            steps = [s - warmup_epochs for s in steps]

            # start from lr * factor and move towards base lr for 'warmup_ratio * epoch' epochs
            warmup = LinearLR(
                optimizer, start_factor=factor, end_factor=1, total_iters=warmup_epochs
            )
            multistep = MultiStepLR(
                optimizer,
                milestones=steps,
                gamma=config.lr_scheduler.gamma,
            )
            return SequentialLR(
                optimizer, [warmup, multistep], milestones=[warmup_epochs]
            )
        else:
            raise ValueError(f"Unknown lr scheduler {config.lr_scheduler.type}")

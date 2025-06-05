import torch
from torch import nn
from transformers.models.upernet.modeling_upernet import UperNetForSemanticSegmentation
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
)
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentation,
)
from transformers import (
    AutoConfig,
    AutoBackbone,
    OneFormerForUniversalSegmentation,
    TimmBackbone,
    TimmBackboneConfig,
)

from models.modules.base import BaseCDEncoder
from models.modules.hf_m2f import handle_m2f_swinb_citysem


class HFBackbone(BaseCDEncoder):
    name = "HFBB"

    def __init__(
        self,
        model_name: str,
        return_layers: list[int],
        pretraining: bool,
        train_args: dict | None = None,
        pretrain_args: dict | None = None,
        lr_adjustments: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            return_layers=return_layers, pretraining=pretraining, *args, **kwargs
        )
        self.model_name = model_name
        self.train_args = train_args
        self.pretrain_args = pretrain_args

        self.finetune: bool
        self.pretrained: bool
        self.backbone: nn.Module | None = None
        self.patch_size: int

        self.lr_adjustments = lr_adjustments  # backwards compatibility
        self.drop_path_rate: bool

        if pretraining:
            self.prepare_for_phase(pretrain_args)
        else:
            self.prepare_for_phase(train_args)

        if self.pretrained:
            print(f"Using pretrained backbone from {model_name}")
            self.name += "PT"

    def prepare_for_phase(self, arg_dict) -> None:
        """
        Set `finetune` and `pretrained` flags, init backbone and freeze it if needed.

        Args:
            arg_dict: dict of currently used args

        Returns:
            None
        """
        self.finetune = arg_dict["finetune"]
        self.pretrained = arg_dict["pretrained"]
        self.drop_path_rate = arg_dict.get("drop_path_rate", None)

        if self.backbone is None:
            self.init_backbone()

        if not self.finetune:
            self.eval()
            self.freeze()
        else:
            self.train()

    def init_backbone(self):
        if self.model_name.startswith("timm"):
            model_name = self.model_name.split(":")[1]
            hf_config = None

            print("Warning: when using timm, return layers are 0-indexed")
            out_list = self.return_layers
        else:
            model_name = self.model_name
            hf_config = AutoConfig.from_pretrained(model_name)
            out_list = [f"stage{i}" for i in self.return_layers]

        if "upernet" in model_name:
            hf_config.backbone_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.backbone_config.drop_path_rate = self.drop_path_rate

            if self.pretrained:
                self.backbone = UperNetForSemanticSegmentation.from_pretrained(
                    model_name, config=hf_config
                ).backbone
            else:
                self.backbone = UperNetForSemanticSegmentation._from_config(
                    config=hf_config
                ).backbone

            self.patch_size = hf_config.backbone_config.patch_size
        elif "mask2former" in model_name:
            hf_config.backbone_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.backbone_config.drop_path_rate = self.drop_path_rate

            if self.pretrained:
                if (
                    model_name
                    == "facebook/mask2former-swin-base-IN21k-cityscapes-semantic"
                ):
                    # fix keys names
                    state_dict = handle_m2f_swinb_citysem(model_name, config=hf_config)
                else:
                    state_dict = None
                self.backbone = Mask2FormerForUniversalSegmentation.from_pretrained(
                    model_name, config=hf_config, state_dict=state_dict
                ).model.pixel_level_module.encoder
            else:
                self.backbone = Mask2FormerForUniversalSegmentation._from_config(
                    config=hf_config
                ).model.pixel_level_module.encoder

            self.patch_size = hf_config.backbone_config.patch_size
        elif "maskformer" in model_name:
            hf_config.backbone_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.backbone_config.drop_path_rate = self.drop_path_rate

            if self.pretrained:
                self.backbone = MaskFormerForInstanceSegmentation.from_pretrained(
                    model_name, config=hf_config
                ).model.pixel_level_module.encoder
            else:
                self.backbone = MaskFormerForInstanceSegmentation._from_config(
                    config=hf_config
                ).model.pixel_level_module.encoder

            if "patch_size" in hf_config.backbone_config:
                self.patch_size = hf_config.backbone_config.patch_size
            else:
                self.patch_size = None
        elif "oneformer" in model_name:
            hf_config.backbone_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.backbone_config.drop_path_rate = self.drop_path_rate

            if self.pretrained:
                self.backbone = OneFormerForUniversalSegmentation.from_pretrained(
                    model_name, config=hf_config
                ).model.pixel_level_module.encoder
            else:
                self.backbone = OneFormerForUniversalSegmentation._from_config(
                    config=hf_config
                ).model.pixel_level_module.encoder

            self.patch_size = hf_config.backbone_config.patch_size
        elif hf_config is None:
            # timm backbones
            if self.pretrained:
                timm_config = TimmBackboneConfig(
                    model_name, use_pretrained_backbone=True
                )
            else:
                timm_config = TimmBackboneConfig(
                    model_name, use_pretrained_backbone=False
                )

            timm_config.out_indices = out_list
            if self.drop_path_rate is not None:
                hf_config.drop_path_rate = self.drop_path_rate

            # dynamic imgs size otherwise exceptions occur
            self.backbone = TimmBackbone(timm_config, dynamic_img_size=True)
            self.patch_size = None
        elif "ImageClassification" in hf_config.architectures[0]:
            # default swin/resnet/etc models
            hf_config.out_features = out_list
            if self.drop_path_rate is not None:
                hf_config.drop_path_rate = self.drop_path_rate

            module = AutoBackbone

            if self.pretrained:
                self.backbone = module.from_pretrained(model_name, config=hf_config)
            else:
                self.backbone = module.from_config(config=hf_config)

            if "patch_size" in hf_config:
                self.patch_size = hf_config.patch_size
            else:
                self.patch_size = None
        else:
            raise ValueError(f"HFBackbone doesn't supports {model_name} models")

        if not self.pretrained and hasattr(self.backbone, "init_weights"):
            self.backbone.init_weights()

    def freeze(self):
        print(f"Freezing backbone")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, mask=None, unnormed_ly3=False):
        if not self.finetune:
            self.backbone.eval()

        if mask is None:
            feat = self.backbone(x)
        else:
            if len(mask.size()) == 3:
                mask = mask.flatten(1)
            feat = self.backbone(x, mask)

        rd = {k: v for k, v in zip(self.return_layers, feat.feature_maps)}

        if unnormed_ly3:
            rd["un_l3"] = feat.unnormed_layer3
        return rd

    def get_parameters(self, pretraining: bool) -> list[dict]:
        if self.finetune:
            # override parent decoder args if phase specifics exists
            if pretraining:
                self.lr = self.pretrain_args.get("lr", self.lr)
                self.weight_decay = self.pretrain_args.get(
                    "weight_decay", self.weight_decay
                )
            else:
                self.lr = self.train_args.get("lr", self.lr)
                self.weight_decay = self.train_args.get(
                    "weight_decay", self.weight_decay
                )
            # get nicely constructed lr & wd dict
            base_lr_dict = self.get_lr_dict()

            if hasattr(self.backbone, "lr_tunable") and self.lr_adjustments:
                print("Adjusting LR for backbone specifics")
                return self.backbone.get_parameters(base_lr_dict)
            else:
                base_params = self.backbone.parameters()
                return [{"params": base_params, **base_lr_dict}]
        else:
            return []

    @torch.no_grad()
    def get_strides(self) -> list[int]:
        was_train = self.backbone.training
        self.backbone.eval()
        feat = self.backbone(torch.rand(1, 3, 256, 256))
        if was_train:
            self.backbone.train()
        return [256 / f.shape[2] for f in feat.feature_maps]

    def get_out_dims(self) -> list[int]:
        return self.backbone.channels

    def transfer_from_pretrained(self, state_dict: dict):
        # currently only swin is properly supported
        state_dict.pop("backbone.embeddings.mask_token")
        super().transfer_from_pretrained(state_dict)

        self.prepare_for_phase(self.train_args)


def build_hfbackbone(config, pretraining):
    return HFBackbone(**config.encoder.init_args.as_dict(), pretraining=pretraining)

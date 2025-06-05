import torch.nn as nn
import torch

from models.modules.base import BaseCDDiff


class SubAbsDiff(BaseCDDiff):
    """
    Simple difference module by subtraction and difference.

    """

    name = "subabs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat_dict):
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        for i, feat in feat_dict.items():
            f1, f2 = feat.chunk(2, dim=0)
            feat_dict[i] = torch.abs(f1 - f2)

        return feat_dict

    def get_parameters(self, pretraining=False):
        return []


class LayerNormPermute(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class SubDiff(BaseCDDiff):
    """
    Simple difference module by subtraction.

    """

    name = "sub"

    def __init__(self, norm: bool = False, dims: list[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if norm:
            self.name += "LN"

        if self.pretraining and not self.use_in_pretrain:
            print("SubDiff won't be used during pretrain")
            return

        self.norms = nn.ModuleList()
        for dim in dims:
            if norm:
                self.norms.append(LayerNormPermute(dim))
            else:
                self.norms.append(nn.Identity())

    def forward(self, feat_dict):
        # skip in pretrain
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        for norm, (i, feat) in zip(self.norms, feat_dict.items()):
            f1, f2 = feat.chunk(2, dim=0)
            feat_dict[i] = norm(f1 - f2)

        return feat_dict

    def get_parameters(self, pretraining=False):
        return [{"params": self.parameters()}]


class CatDiff(BaseCDDiff):
    """
    Simple difference module by concatenation.

    """

    name = "cat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat_dict):
        raise NotImplementedError("refactor before use")
        for i, feat in feat_dict.items():
            f1, f2 = feat.chunk(2, dim=0)
            # channel cat
            feat_dict[i] = torch.cat((f1, f2), dim=1)

        return feat_dict

    def adjust_dims(self, feat_dims: list[int]) -> list[int]:
        """
        Adjust output dims for concat style - by doubling.

        Args:
            feat_dims: list of input feature dims

        Returns:
            list of updated feature dims based on diff style
        """
        return [d * 2 for d in feat_dims]

    def get_parameters(self, pretraining=False) -> list[dict]:
        return []

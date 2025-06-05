from torch import nn

from models.modules.base import BaseCDEncoder
from torchvision.models import resnet50
import torch
from torchvision.models.feature_extraction import create_feature_extractor


class CacoRN50(BaseCDEncoder):
    name = "CaCo"

    def __init__(self, pt_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rn50 = resnet50()
        state = torch.load(pt_path, map_location=torch.device("cpu"))

        state = rename_keys(rn50.state_dict(), state)

        msg = rn50.load_state_dict(state, strict=False)
        print(msg)
        self.fe = create_feature_extractor(
            rn50, return_nodes=[f"layer{i}" for i in self.return_layers]
        )

    def forward(self, x):
        return self.fe(x)

    def get_parameters(self, pretraining: bool) -> list[dict]:
        return [{"params": self.fe.parameters()}]

    @torch.no_grad()
    def get_out_dims(self):
        self.eval()
        features = self.forward(torch.rand(1, 3, 256, 256))
        return [f.shape[1] for f in features.values()]

    @torch.no_grad()
    def get_strides(self):
        self.eval()
        features = self.forward(torch.rand(1, 3, 256, 256))
        return [256 / f.shape[2] for f in features.values()]


def rename_keys(current_state, loaded_state_dict):
    renamed_state_dict = {}
    for current_key, (l_k, l_v) in zip(current_state.keys(), loaded_state_dict.items()):
        ck_p = current_key.split(".")[1:]  # layer1.0.bn1.weight -> 0.bn1.weight
        lk_p = l_k.split(".")[1:]  # 4.0.bn1.weight -> 0.bn1.weight
        if ck_p == lk_p:
            renamed_state_dict[current_key] = l_v
        else:
            raise ValueError(f"missmatch: {ck_p} != {lk_p}")

    return renamed_state_dict


def build_caco(config):
    return CacoRN50(pretraining=True, **config.encoder.init_args.as_dict())

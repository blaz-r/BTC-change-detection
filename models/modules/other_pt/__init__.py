from models.modules.other_pt.gfm import build_gfm

from .base import OtherPT
from .caco import build_caco
from .gassl import build_gassl
from .mtp import vit_b_rvsa
from .satmae import satmae_vit_base_patch16
from .seco import build_seco


def build_other_pt(config):
    pt_name = config.encoder.init_args.pt_name

    if pt_name == "gfm":
        return build_gfm(config)
    elif pt_name == "mtp":
        return vit_b_rvsa(config)
    elif pt_name == "satmae":
        return satmae_vit_base_patch16(config)
    elif pt_name == "caco_rn50":
        return build_caco(config)
    elif pt_name == "seco_rn50":
        return build_seco(config)
    elif pt_name == "gassl":
        return build_gassl(config)

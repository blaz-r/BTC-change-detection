from abc import ABC

from torch import nn


class BaseCDModule(nn.Module, ABC):
    """
    Base class for all change detection.

    Args:
        pretraining (bool): flag if module is in pretraining state
        transfer (bool): transfer to finetuning phase or reinitialize
        lr (float): learning rate
        weight_decay (float): weight decay

    """

    def __init__(
        self,
        pretraining: bool,
        transfer: bool = True,
        lr: float | None = None,
        weight_decay: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretraining = pretraining
        self.transfer = transfer
        self.lr = lr
        self.weight_decay = weight_decay

    def get_lr_dict(self):
        out_dict = {}
        if self.lr is not None:
            out_dict["lr"] = self.lr
        if self.weight_decay is not None:
            out_dict["weight_decay"] = self.weight_decay

        if out_dict:
            print(
                f">>>Overriding {self.__class__.__name__} opti. settings with {out_dict}"
            )

        return out_dict

    def get_parameters(self, pretraining: bool) -> list[dict]:
        raise NotImplementedError

    def transfer_from_pretrained(self, pretrained_state_dict: dict) -> None:
        """
        Handle transfer of state dict and change of parameters on transfer.

        Default just sets current state dict ot passed state dict

        Returns:
            None
        """
        missing, unexpected = self.load_state_dict(pretrained_state_dict, strict=False)
        print(f"Missing : {missing}, Unexpected : {unexpected}")


class BaseCDEncoder(BaseCDModule):
    def __init__(self, return_layers=list[int], *args, **kwargs):
        """
        Base encoder module, must accept return_layers

        Args:
            return_layers: list of layer IDs to be returned. If -1, return all
            *args: args of subclass
            **kwargs: kwargs of subclass
        """
        super().__init__(*args, **kwargs)
        self.return_layers = return_layers

    def get_out_dims(self):
        raise NotImplementedError

    def get_strides(self):
        raise NotImplementedError


class BaseCDDiff(BaseCDModule):
    def __init__(self, use_in_pretrain: bool = False, *args, **kwargs) -> None:
        self.use_in_pretrain = use_in_pretrain
        super().__init__(*args, **kwargs)

    def adjust_dims(self, feat_dims: list[int]) -> list[int]:
        """
        Adjust output dims based on diff style.

        Args:
            feat_dims: list of input feature dims

        Returns:
            list of updated feature dims based on diff style
        """
        # default - no change
        return feat_dims


class BaseCDDecoder(BaseCDModule):
    def __init__(
        self,
        input_sizes: list[int],
        encoder_strides: list[int],
        out_size: int = None,
        out_channels: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.out_size = out_size
        self.input_sizes = input_sizes
        self.encoder_strides = encoder_strides
        self.out_channels = out_channels


class BaseCDOutProc(BaseCDModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class BaseCDPreDiff(BaseCDModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

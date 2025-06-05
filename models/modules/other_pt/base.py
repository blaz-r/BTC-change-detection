from models.modules.base import BaseCDEncoder


class OtherPT(BaseCDEncoder):
    def __init__(
        self,
        pt_name: str,
        pt_path: str,
        window_size: int = None,
        return_down: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

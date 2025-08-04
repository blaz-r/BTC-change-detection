from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data.dataset import CDDataset
from data.transforms import build_transforms

from data.hf_datasets import (
    get_levircd,
    get_sysu,
    get_egybcd,
    get_clcd,
    get_gvlm,
    get_oscd96,
)


class CDDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        pretrain: bool,
        data_path: str = None,
        use_hf=True,
        load_in_mem=False,
    ):
        super().__init__()

        self.config = config
        self.data_path = Path(data_path)
        self.pretrain = pretrain
        self.use_hf = use_hf
        self.load_in_mem = load_in_mem

        self.batch_size = config.data.batch_size

        self.prepare_dataset()

        print(f"Num workers: {config.data.num_workers}")

    def prepare_dataset(self):
        if self.pretrain:
            self.dataset_name = self.config.data.pretrain_dataset
        else:
            self.dataset_name = self.config.data.dataset

        if self.use_hf:
            data = get_hf_dataset(self.dataset_name, self.data_path)
            data.set_format("numpy")
        else:
            data = {
                "train": self.data_path / self.dataset_name / "train",
                "test": self.data_path / self.dataset_name / "test",
                "val": self.data_path / self.dataset_name / "val",
            }

        train_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=False
        )
        test_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=True
        )

        self.train_data = CDDataset(
            data["train"],
            train_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
        )
        self.test_data = CDDataset(
            data["test"],
            test_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
        )

        if (
            (self.use_hf and "val" in data)
            or (self.load_in_mem == "direct" and data["val"].exists())
            or (self.load_in_mem == "hdf5" and Path(str(data["val"]) + ".h5").exists())
        ):
            self.val_data = CDDataset(
                data["val"],
                test_transforms,
                use_hf=self.use_hf,
                load_in_mem=self.load_in_mem,
            )
        else:
            print("Validation data not present, using test set.")
            self.val_data = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            drop_last=True,
            shuffle=True,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
        )


def get_hf_dataset(name, data_path):
    if name == "oscd96":
        ds = get_oscd96(data_path)
    elif name == "levir":
        ds = get_levircd(data_path)
    elif name == "sysu":
        ds = get_sysu(data_path)
    elif name == "egybcd":
        ds = get_egybcd(data_path)
    elif name == "clcd":
        ds = get_clcd(data_path)
    elif name == "gvlm":
        ds = get_gvlm(data_path)
    else:
        msg = (
            f"Unknown HuggingFace dataset {name}. If you want to use directory or hdf5 dataset, set `data.use_hf` to False."
            f"Optionally set `data.load_in_mem` to 'direct' or 'hdf5'"
        )
        raise ValueError(msg)

    for key in ds:
        ds[key] = ds[key].add_column("img_idx", range(len(ds[key])))

    return ds

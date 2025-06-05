from pathlib import Path

import h5py
import timeit
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import datasets as hf
from tqdm import tqdm


class CDDataset(Dataset):
    def __init__(
        self, data: str | list[str], transform, use_hf=True, load_in_mem="direct"
    ):
        self.transform = transform
        self.use_hf = use_hf
        self.load_in_mem = load_in_mem

        if use_hf:
            print("Using HF data")
            self.data = data  # self.load_from_hf_dataset(data)
        else:
            if load_in_mem == "direct":
                self.data = self.load_from_dir(data)
            elif load_in_mem == "hdf5":
                if isinstance(data, list):
                    print("loading multiple datasets")
                    arr = []
                    for d in data:
                        arr.extend(self.load_from_hdf5(d))
                    self.data = arr
                else:
                    self.data = self.load_from_hdf5(data)

            else:
                self.data = self.load_paths_from_dir(data)

    def load_from_hf_dataset(self, data: hf.Dataset):
        """
        Load to memory to avoid issues. (Still works poorly)

        Args:
            data: hf dataset

        Returns:
            data in list
        """
        data_list = []

        print("loading from HF dataset")
        for batch in tqdm(data.iter(batch_size=42), total=len(data) / 42):
            for i in range(len(batch["imageA"])):
                t_el = {
                    "imageA": batch["imageA"][i],
                    "imageB": batch["imageB"][i],
                    "label": batch["label"][i],
                    "img_idx": batch["img_idx"][i],
                }
                data_list.append(t_el)

        return data_list

    def load_paths_from_dir(self, path):
        path = Path(path)

        file_list = list(Path(path).glob("A/*.png"))
        if len(file_list) == 0:
            raise ValueError(f"No images found in {path}")

        file_dict_list = []
        for img_path in tqdm(
            file_list, desc="Loading paths from dir.", total=len(file_list)
        ):
            fname = Path(img_path).name
            t_el = {
                "imageA": img_path,
                "imageB": path / "B" / fname,
                "label": path / "label" / fname,
                "img_idx": img_path.stem,
            }
            file_dict_list.append(t_el)

        return file_dict_list

    def load_from_dir(self, path):
        path = Path(path)

        file_list = list(Path(path).glob("A/*"))
        if len(file_list) == 0:
            raise ValueError(f"No images found in {path}")

        data_list = []
        for img_path in tqdm(
            file_list, desc="Loading images into memory.", total=len(file_list)
        ):
            fname = Path(img_path).name
            t_el = {
                "imageA": np.array(Image.open(img_path)),
                "imageB": np.array(Image.open(path / "B" / fname)),
                "label": np.array(Image.open(path / "label" / f"{img_path.stem}.png")),
                "img_idx": img_path.stem,
            }
            data_list.append(t_el)

        return data_list

    def load_imgs_from_path(self, path_dict):
        res_dict = {
            "imageA": np.array(Image.open(path_dict["imageA"])),
            "imageB": np.array(Image.open(path_dict["imageB"])),
            "label": np.array(Image.open(path_dict["label"])),
            "img_idx": path_dict["img_idx"],
        }
        return res_dict

    def load_from_hdf5(self, path):
        d_name = Path(path).parts[-2]
        split = Path(path).parts[-1]
        print(f"Loading from HDF5 {d_name} {split}")
        t1 = timeit.default_timer()
        file = h5py.File(str(path) + ".h5", "r")

        imageA = np.array(file["/imageA"])
        imageB = np.array(file["/imageB"])
        label = np.array(file["/label"])
        img_idx = np.array(file["/img_idx"])

        file.close()
        t2 = timeit.default_timer()
        print(f"Done in {t2 - t1:.2f} seconds")

        ret_list = []
        for a, b, lbl, idx in zip(imageA, imageB, label, img_idx):
            ret_list.append(
                {
                    "imageA": a,
                    "imageB": b,
                    "label": lbl,
                    "img_idx": idx,
                    "d_name": d_name,
                }
            )

        return ret_list

    def __getitem__(self, index):
        data = self.data[index]

        if self.use_hf:
            # remove writeable flag
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = np.copy(v)
        elif not self.load_in_mem:
            data = self.load_imgs_from_path(data)

        transformed = self.transform(data)
        # save unnomred for visualiser
        transformed["imageA_unnorm"] = data["imageA"]
        transformed["imageB_unnorm"] = data["imageB"]
        transformed["img_idx"] = data["img_idx"]
        if "d_name" in data:
            transformed["d_name"] = data["d_name"]

        return transformed

    def __len__(self):
        return len(self.data)

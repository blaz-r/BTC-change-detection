from datasets import load_dataset


def get_levircd(path):
    ds = load_dataset("ericyu/LEVIRCD_Cropped256", cache_dir=path)
    return ds


def get_sysu(path):
    ds = load_dataset("ericyu/SYSU_CD", cache_dir=path)
    return ds


def get_egybcd(path):
    ds = load_dataset("ericyu/EGY_BCD", cache_dir=path)
    return ds


def get_clcd(path):
    ds = load_dataset("ericyu/CLCD_Cropped_256", cache_dir=path)
    return ds


def get_gvlm(path):
    ds = load_dataset("ericyu/GVLM_Cropped_256", cache_dir=path)
    return ds


def get_oscd96(path):
    ds = load_dataset("blaz-r/OSCD_RGB_Cropped_96", cache_dir=path)
    return ds

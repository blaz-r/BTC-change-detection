import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
import timeit

import sys

sys.path.append("..")

from configs.config_parser import get_parser
from models.finetune_framework import FinetuneFramework


def params(model):
    np = list((n.split(".")[0], p.numel()) for n, p in model.named_parameters())
    collected = defaultdict(int)
    for n, p in np:
        collected[n] += p

    collected["total"] = sum(collected.values())

    collected = {k: v / 10**6 for k, v in collected.items()}
    return collected


@torch.no_grad()
def flops(model):
    img1 = torch.randn(1, 3, 256, 256, dtype=torch.float16, device="cuda")
    img2 = torch.randn(1, 3, 256, 256, dtype=torch.float16, device="cuda")

    batch = {"imageA": img1, "imageB": img2}
    model.cuda()
    model.eval()

    # first - warmup
    out = model(batch)

    # real - don't need reps as the result is always same
    with torch.profiler.profile(with_flops=True) as prof:
        out = model(batch)
    tflops = sum(x.flops for x in prof.key_averages()) / 1e9

    return tflops


@torch.no_grad()
def inference_speed(model, reps=1000):
    img1 = torch.randn(1, 3, 256, 256, dtype=torch.float16, device="cuda")
    img2 = torch.randn(1, 3, 256, 256, dtype=torch.float16, device="cuda")

    batch = {"imageA": img1, "imageB": img2}
    model.cuda()
    model.eval()

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        out = model(batch)

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Timing inference"):
        t0 = timeit.default_timer()

        out = model(batch)

        t1 = timeit.default_timer()
        total_time += t1 - t0

    # * 1000 to get ms
    ms = total_time * 1000 / reps
    return ms


def prepare_model(path):
    parser = get_parser()
    config = parser.parse_args(["--config", path])

    try:
        # handle RSFM weights path
        old_p = config["encoder"]["init_args"]["pt_path"]
        config["encoder"]["init_args"]["pt_path"] = f"../{old_p}"
    except:
        pass

    model = FinetuneFramework(
        config_namespace=config,
        config=config.as_dict(),
        metrics=None,
        logger=None,
    )

    model.eval()
    model.to(torch.float16)
    model.cuda()
    return model


def get_models():
    configs = {}
    configs["our_b"] = "../configs/exp/BTC-B.yaml"
    configs["our_t"] = "../configs/exp/BTC-T.yaml"
    other = {
        x: f"../configs/exp/sota/other/{x}.yaml"
        for x in [
            "gfm_swinB",
            "mtp_vitBUnet",
            "satmae_vitlUnet",
            "caco_rn50",
            "seco_rn50",
            "gassl",
        ]
    }
    configs = {**configs, **other}

    models = {}

    for m, p in configs.items():
        model = prepare_model(p)
        models[m] = model

    return models


def main():
    models = {"btc-t": prepare_model("../configs/exp/BTC-T.yaml")}
    perf_dict = {}
    for m in models:
        curr_d = {}
        pdict = params(models[m])
        curr_d["params"] = pdict["total"]
        curr_d["enc_p"] = pdict["enc"]
        curr_d["dec_p"] = pdict["dec"]

        curr_d["flops"] = flops(models[m])
        time_arr = []
        repeat = 6
        for repetas in range(repeat):
            ct = inference_speed(models[m], reps=1000)
            time_arr.append(ct)
        # skip first - additional warmup
        inf_t = sum(time_arr[1:]) / (repeat - 1)

        # / 1000 to get to s, then inverse to get fps
        fps = 1 / (inf_t / 1000)
        curr_d["fps"] = fps
        curr_d["time"] = inf_t
        curr_d["time_arr"] = time_arr[1:]

        perf_dict[m] = curr_d

    with open("btc_perf.json", "w") as f:
        json.dump(perf_dict, f)


if __name__ == "__main__":
    main()

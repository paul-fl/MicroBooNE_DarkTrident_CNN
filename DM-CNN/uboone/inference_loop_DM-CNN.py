import os, sys, ROOT
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary
from lib.config import config_loader
from lib.utility import get_fname


def run_single_inference(cfg, root_file, csv_file, weight_file, output_dir):
    print("\n======================================")
    print("Running inference:")
    print("ROOT  :", root_file)
    print("CSV   :", csv_file)
    print("WEIGHT:", weight_file)
    print("OUTDIR:", output_dir)
    print("======================================\n")

    df = pd.read_csv(csv_file)
    df["signal_score"] = -999
    df["entry_number"] = -1
    df["n_pixels"]     = -1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mpid = mpid_net_binary.MPID().to(device)
    mpid.load_state_dict(torch.load(weight_file, map_location=device))
    mpid.eval()

    test_data = mpid_data_binary.MPID_Dataset(
        root_file, 
        "image2d_image2d_binary_tree", 
        device
    )
    n_events = test_data[0][3]

    print(f"Total events: {n_events}")
    print("Starting inference...")

    start = time.time()

    for ENTRY in range(n_events - 1):
        if ENTRY % 1000 == 0:
            print("ENTRY:", ENTRY)

        run_info, subrun_info, event_info = test_data[ENTRY][2]
        index_array = df.query(
            f"run_number == {run_info} & subrun_number == {subrun_info} & event_number == {event_info}"
        ).index.values

        input_image = test_data[ENTRY][0].view(-1, 1, 512, 512)

        input_image[0][0][input_image[0][0] > cfg.adc_hi] = cfg.adc_hi
        input_image[0][0][input_image[0][0] < cfg.adc_lo] = 0

        score = nn.Sigmoid()(mpid(input_image.to(device))).cpu().detach().numpy()[0]

        if len(index_array) == 0:
            continue

        df.loc[index_array[0], "signal_score"] = score[0]
        df.loc[index_array[0], "entry_number"] = ENTRY
        df.loc[index_array[0], "n_pixels"]     = np.count_nonzero(input_image)

    elapsed = time.time() - start
    print(f"Inference time: {elapsed:.4f} seconds")

    os.makedirs(output_dir, exist_ok=True)
    out_csv = f"{output_dir}/{get_fname(root_file)}_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV → {out_csv}")

    plt.figure()
    dp = df[df["signal_score"] >= 0]
    plt.hist(dp["signal_score"], bins=40, alpha=0.9)
    plt.xlabel("Signal Score")
    plt.grid()
    plt.savefig(f"{output_dir}/{get_fname(root_file)}_scores.png")
    plt.savefig(f"{output_dir}/{get_fname(root_file)}_scores.pdf")


def InferenceCNN():
    # ⭐ HARD-CODED config path (no args, no fallback)
    CFG = "/workspace/DM-CNN/cfg/inference_loop.cfg"
    print("Using config:", CFG)
    cfg = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)

    # comma-separated lists → Python lists
    weight_list = [x.strip() for x in cfg.weights.split(",")]
    root_files  = [x.strip() for x in cfg.input_files.split(",")]
    csv_files   = [x.strip() for x in cfg.input_csvs.split(",")]

    print("\n======== CONFIG LOADED ========")
    print("Weights:", weight_list)
    print("ROOTs  :", root_files)
    print("CSVs   :", csv_files)
    print("Outdir :", cfg.output_base)
    print("================================\n")

    # main loop
    for weight_file in weight_list:
        for root_file, csv_file in zip(root_files, csv_files):

            tag_base = get_fname(root_file)
            tag_wt   = get_fname(weight_file)

            out_dir = os.path.join(
                cfg.output_base,
                f"{tag_base}__{tag_wt}"
            )

            run_single_inference(cfg, root_file, csv_file, weight_file, out_dir)


if __name__ == "__main__":
    InferenceCNN()



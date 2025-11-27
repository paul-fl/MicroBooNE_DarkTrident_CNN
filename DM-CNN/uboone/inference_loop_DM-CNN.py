# inference_loop_DM-CNN.py
import os, sys, ROOT
import getopt, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# MPID modules
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary
from lib.config import config_loader
from lib.utility import get_fname

def InferenceCNN():
    """
    Perform inference using a trained DM-CNN model,
    reading parameters from a .cfg file passed with --cfg
    """

    # ✅ NEW: accept --cfg argument dynamically
    if len(sys.argv) > 2 and sys.argv[1] == "--cfg":
        CFG = sys.argv[2]
    else:
        MPID_PATH = os.path.dirname(mpid_data_binary.__file__) + "/../cfg"
        CFG = os.path.join(MPID_PATH, "inference_config_binary.cfg")

    print(f"Loading config from: {CFG}")
    cfg = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUID
    train_device = "cuda" if torch.cuda.is_available() else "cpu"

    input_file = cfg.input_file
    input_csv = cfg.input_csv
    output_dir = cfg.output_dir
    tag = cfg.name
    weight_file = cfg.weight_file
    rotate = cfg.rotation

    # Build output file name
    file_name = get_fname(input_file)
    output_file = f"{output_dir}{file_name}_DM-CNN_scores_{tag}.csv"

    print("\nRunning DM-CNN inference...")
    print("Input larcv:", input_file)
    print("Weights:", weight_file)
    print("Output CSV:", output_file)
    print("Rotation:", rotate)
    print()

    df = pd.read_csv(input_csv)
    df["signal_score"] = np.ones(len(df)) * -999999.9
    df["entry_number"] = np.ones(len(df)) * -1
    df["n_pixels"] = np.ones(len(df)) * -1

    mpid = mpid_net_binary.MPID()
    mpid.cuda()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()

    test_data = mpid_data_binary.MPID_Dataset(input_file, "image2d_image2d_binary_tree", train_device)
    n_events = test_data[0][3]

    print("Total number of events:", n_events)
    print("Starting inference...")

    init = time.time()
    for ENTRY in range(n_events - 1):
        if ENTRY % 1000 == 0:
            print("ENTRY:", ENTRY)

        run_info, subrun_info, event_info = test_data[ENTRY][2]
        index_array = df.query(
            f"run_number == {run_info} & subrun_number == {subrun_info} & event_number == {event_info}"
        ).index.values

        input_image = test_data[ENTRY][0].view(-1, 1, 512, 512)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10] = 0

        if rotate:
            from scipy.ndimage import rotate as imrotate
            input_image[0][0] = torch.tensor(imrotate(input_image[0][0], angle=rotation_angle))

        score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]

        if len(index_array) == 0:
            continue

        df.loc[index_array[0], "signal_score"] = score[0]
        df.loc[index_array[0], "entry_number"] = ENTRY
        df.loc[index_array[0], "n_pixels"] = np.count_nonzero(input_image)

    end = time.time()
    print(f"Total processing time: {end - init:.4f} seconds")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to: {output_file}")

    plt.figure()
    dp = df[df["signal_score"] >= 0.0]
    plt.hist(dp["signal_score"], bins=40, alpha=0.9, label=file_name, histtype="bar")
    plt.xlabel("Signal score")
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir}{file_name}_DM-CNN_signal_score_distribution_{tag}.png")
    plt.savefig(f"{output_dir}{file_name}_DM-CNN_signal_score_distribution_{tag}.pdf")

    return 0


if __name__ == "__main__":
    InferenceCNN()


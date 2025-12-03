import os, sys, ROOT, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from mpid_data import mpid_data_multiclass
from mpid_net import mpid_net_multiclass
from lib.config import config_loader
from lib.utility import get_fname


def run_single_inference(cfg, root_file, csv_file, weight_file, output_dir):

    df = pd.read_csv(csv_file)
    df["p0"] = -999
    df["p1"] = -999
    df["p2"] = -999
    df["p3"] = -999
    df["p4"] = -999
    df["pred_class"] = -1
    df["true_class"] = -1
    df["entry_number"] = -1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mpid_net_multiclass.MPID(dropout=0.2, num_classes=5).to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    test_data = mpid_data_multiclass.MPID_Dataset(
        root_file,
        "image2d_image2d_binary_tree",
        device
    )
    n_events = test_data[0][3]

    start = time.time()

    for ENTRY in range(n_events - 1):

        run_info, subrun_info, event_info = test_data[ENTRY][2]
        index_array = df.query(
            f"run_number == {run_info} & subrun_number == {subrun_info} & event_number == {event_info}"
        ).index.values

        if len(index_array) == 0:
            continue

        x = test_data[ENTRY][0].view(-1, 1, 512, 512).to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()[0]

        pred_class = int(np.argmax(probs))
        true_class = int(test_data[ENTRY][1])

        idx = index_array[0]
        df.loc[idx, ["p0","p1","p2","p3","p4"]] = probs
        df.loc[idx, "pred_class"] = pred_class
        df.loc[idx, "true_class"] = true_class
        df.loc[idx, "entry_number"] = ENTRY

    elapsed = time.time() - start

    os.makedirs(output_dir, exist_ok=True)
    out_csv = f"{output_dir}/{get_fname(root_file)}_scores_multiclass.csv"
    df.to_csv(out_csv, index=False)

    return df


def InferenceCNN():

    CFG = "/workspace/DM-CNN/cfg/inference_multiclass.cfg"
    cfg = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)

    weight_list = [x.strip() for x in cfg.weights.split(",")]
    root_files  = [x.strip() for x in cfg.input_files.split(",")]
    csv_files   = [x.strip() for x in cfg.input_csvs.split(",")]

    for weight_file in weight_list:
        for root_file, csv_file in zip(root_files, csv_files):

            tag_base = get_fname(root_file)
            tag_wt   = get_fname(weight_file)
            out_dir = os.path.join(cfg.output_base, f"{tag_base}__{tag_wt}")

            df = run_single_inference(cfg, root_file, csv_file, weight_file, out_dir)

            cm = pd.crosstab(df["true_class"], df["pred_class"])

            plt.figure(figsize=(6,5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(5))
            plt.yticks(range(5))
            plt.title("Confusion Matrix")
            plt.savefig(f"{out_dir}/confusion_matrix.png")
            plt.savefig(f"{out_dir}/confusion_matrix.pdf")
            plt.close()


if __name__ == "__main__":
    InferenceCNN()


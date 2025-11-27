#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# === PATHS (REAL HOST PATHS) ===
base_dir = "/home/hep/pf522/dark_trident_wspace/outputs/inference_results"
steps = ["7346", "8441", "9821"]

# choose signal and background datasets
signal_file = "dm_signal_only_test_set_DM-CNN_scores_inference_DM_CNN_pf522.csv"
background_file = "cosmics_corsika_test_set_DM-CNN_scores_inference_DM_CNN_pf522.csv"

# output plot directory
plot_outdir = os.path.join(base_dir, "ROC_plots")
os.makedirs(plot_outdir, exist_ok=True)

# --- helper function to load scores ---
def load_scores(path, label):
    df = pd.read_csv(path)
    if "signal_score" not in df.columns:
        raise ValueError(f"Missing 'signal_score' column in {path}")
    df["label"] = label
    return df[["signal_score", "label"]]

# --- plot setup ---
plt.figure(figsize=(8, 6))

# --- loop through all training steps ---
for step in steps:
    step_dir = os.path.join(base_dir, f"step_{step}")
    sig_path = os.path.join(step_dir, signal_file)
    bkg_path = os.path.join(step_dir, background_file)

    if not (os.path.exists(sig_path) and os.path.exists(bkg_path)):
        print(f"⚠️ Missing data for step {step}, skipping.")
        continue

    # load data
    df_sig = load_scores(sig_path, 1)
    df_bkg = load_scores(bkg_path, 0)
    df_all = pd.concat([df_sig, df_bkg], ignore_index=True)

    # compute ROC
    fpr, tpr, _ = roc_curve(df_all["label"], df_all["signal_score"])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"Step {step} (AUC = {roc_auc:.3f})")

# --- styling ---
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate (Background acceptance)")
plt.ylabel("True Positive Rate (Signal efficiency)")
plt.title("DM-CNN ROC Curves across Training Steps")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# --- save plot ---
plot_path = os.path.join(plot_outdir, "DM_CNN_ROC_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"\n✅ ROC curves saved to: {plot_path}\n")


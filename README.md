# MicroBooNE Dark Trident CNN

Convolutional neural network to separate **dark trident** (dark-matter) interactions from neutrino and cosmic-ray backgrounds in MicroBooNE liquid-argon TPC images (512 x 512). Part of my MSci project at Imperial College London (2025-26).

## What's here
- `DM-CNN/`: the DM-CNN classifier, built on the MPID network. Based on [lmlepin9/DM-CNN](https://github.com/lmlepin9/DM-CNN). Binary and multiclass training, inference and occlusion analysis.
- `scripts/`: training curves and ROC curves.
- `*.job`, `*_payload.sh`: batch jobs for training, inference, occlusion and image printing on a GPU cluster.

## What I did
This was a two-person project. My contributions:
- **Multiclass classifier**: extended the binary DM-CNN to separate signal from several background classes.
- **Occlusion analysis**: studied which image regions drive the network's decisions.
- **Cluster pipeline**: batch jobs for training, inference, occlusion and event-image printing on a GPU cluster.

My project partner developed the binary CNN.

Note: MicroBooNE data and results are collaboration property; this repo contains code only.

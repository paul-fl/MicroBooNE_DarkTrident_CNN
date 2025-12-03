#!/bin/bash

# Go to your workspace
cd /home/hep/pf522/dark_trident_wspace

# Run inside Apptainer with GPU and with both workspace + larcv_files bound
apptainer exec --nv \
  -B /home/hep/pf522/dark_trident_wspace:/workspace \
  -B /vols/sbn/uboone/darkTridents/data/larcv_files:/larcv_files \
  /vols/sbn/uboone/pf522/docker_img/larcv2_py3_1.1.sif \
  bash -c "
cd /workspace/DM-CNN
source setup_larcv2_dm.sh

# Run inference loop (cfg loaded automatically inside script)
python3 ./uboone/inference_multiclass.py
"


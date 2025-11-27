#!/bin/bash
cd /home/hep/pf522/dark_trident_wspace

apptainer exec --nv -B $PWD:/workspace /vols/sbn/uboone/pf522/docker_img/larcv2_py3_1.1.sif /bin/bash -c "
cd /workspace/DM-CNN
source setup_larcv2_dm.sh
python3 ./uboone/print_image_with_score.py -n 0
"


#!/bin/bash
# inference_loop_payload.sh — run inference for multiple training steps and datasets

cd /home/hep/pf522/dark_trident_wspace

# Apptainer environment
apptainer exec --nv -B $PWD:/workspace /vols/sbn/uboone/pf522/docker_img/larcv2_py3_1.1.sif /bin/bash -s <<'EOF'

cd /workspace/DM-CNN
source setup_larcv2_dm.sh

# --- Define which checkpoints and datasets to loop over ---
steps=(7346 8441 9821)
declare -A datasets=(
  ["dm_signal_only"]="dm_signal_only_test_set.root"
  ["cosmics_corsika"]="cosmics_corsika_test_set.root"
  ["ncpio_only"]="ncpi0_only_test_set.root"
  ["ncpio_corsika"]="ncpi0_corsika_test_set.root"
)

# --- Loop over steps and datasets ---
for step in "${steps[@]}"; do
  weight_path=$(ls /workspace/CNN_weights/*step_${step}.pwf 2>/dev/null | head -n1)
  if [ -z "$weight_path" ]; then
    echo "❌ No weight file found for step ${step}"
    continue
  fi

  echo "=== Running inference for step ${step} ==="

  for file_key in "${!datasets[@]}"; do
    input_root="/workspace/cnn_datasets/${datasets[$file_key]}"
    input_csv="/workspace/cnn_datasets/${file_key}.csv"

    if [ ! -f "$input_root" ]; then
      echo "⚠️ Missing ROOT file: $input_root"
      continue
    fi
    if [ ! -f "$input_csv" ]; then
      echo "⚠️ Missing CSV: $input_csv"
      continue
    fi

    echo "→ Dataset: $file_key (step $step)"
    output_dir="/workspace/outputs/inference_results/step_${step}/"
    mkdir -p "$output_dir"

    cfg_dir="/workspace/DM-CNN/cfg/temp"
    mkdir -p "$cfg_dir"
    cfg_path="${cfg_dir}/tmp_cfg_${file_key}_step_${step}.cfg"

    cat <<CFG > "$cfg_path"
adc_lo = float(10)
adc_hi = float(500)
GPUID = str("0")
name = str("inference_DM_CNN_pf522")
rotation = False

input_file = "${input_root}"
input_csv  = "${input_csv}"
weight_file = "${weight_path}"
output_dir = "${output_dir}"
CFG

    echo "   → Config: $cfg_path"
    python3 ./uboone/inference_loop_DM-CNN.py --cfg "$cfg_path"
  done
done

EOF




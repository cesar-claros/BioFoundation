#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* LuMamba side of the same-windows LuMamba-vs-HYDRA comparison.
#*
#* For each window length in {15,30,45,60}s and seed in {0..4}: regenerate the subject split,
#* SAVE the window manifests (windows_{train,val,test}.csv -> $MANIFEST_ROOT/w<ws>_s<seed>/) so
#* HYDRA can crop the exact same time-segments, full-finetune all 4 LuMamba variants, and eval
#* with subject-level threshold calibration on val. Results append to $RESULTS_CSV (resumable).
#*
#* EDIT ROOT_DIR, then from the repo root:
#*   nohup bash run_lumamba_vs_hydra.sh > lumamba_vs_hydra.log 2>&1 &
#* Override inline, e.g.:  WINDOWS="30 60" SEEDS="0 1 2" bash run_lumamba_vs_hydra.sh
#*
#* WARNING: regenerating overwrites <DATA_PATH>/TUEP_data/*.h5 each build (the manifests persist).
#* Full grid = 4 lengths x 5 seeds x 4 variants = 80 finetunes (~3-5 h).
#*----------------------------------------------------------------------------*
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data/v3.0.0}"  # <-- EDIT: dir with 00_epilepsy/ 01_no_epilepsy/
export DATA_PATH="${DATA_PATH:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/BioFoundation}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/checkpoints}"

WINDOWS="${WINDOWS:-15 30 45 60}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LR="${LR:-1e-4}"
RESULTS_CSV="${RESULTS_CSV:-lumamba_vs_hydra_results.csv}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$DATA_PATH/manifests}"

cd "$(dirname "$0")"
[ -d "$ROOT_DIR" ] || { echo "ERROR: ROOT_DIR not found: $ROOT_DIR (edit run_lumamba_vs_hydra.sh)"; exit 1; }

echo "ROOT_DIR=$ROOT_DIR | DATA_PATH=$DATA_PATH | CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "windows=[$WINDOWS] seeds=[$SEEDS] lr=$LR -> $RESULTS_CSV | manifests -> $MANIFEST_ROOT"

for WS in $WINDOWS; do
    # Memory scales ~linearly with window length (Mamba); scale the batch inversely so 60 s does
    # not OOM in training (15s->128, 30s->64, 45s->42, 60s->32). Override with BATCH_SIZE=... .
    if [ -n "${BATCH_SIZE:-}" ]; then BW="$BATCH_SIZE"; else BW=$(( 1920 / WS )); fi
    echo "==================== LuMamba window_s=${WS}s  batch=${BW} ===================="
    python -u sweep_foundation_models.py --root_dir "$ROOT_DIR" \
        --window_s "$WS" --seeds $SEEDS \
        --variants reconstruction_only lejepa_only_128 mixed_128 mixed_300 --modes full \
        --batch_size "$BW" --lr "$LR" --calib_split val \
        --manifest_root "$MANIFEST_ROOT" --results_csv "$RESULTS_CSV"
done

echo "LuMamba sweep done. Manifests for HYDRA are under $MANIFEST_ROOT/w<ws>_s<seed>/."

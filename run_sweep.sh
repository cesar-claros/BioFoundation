#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Multi-seed sweep of the pretrained foundation models on TUEP epilepsy diagnosis.
#* Drives sweep_foundation_models.py over seeds x {frozen, full} x the 4 LuMamba variants,
#* regenerating the subject-level split per seed and collecting subject/window metrics into
#* a CSV with a mean +/- std summary. Resumable: cells already in the CSV are skipped.
#*
#* EDIT ROOT_DIR below (your TUH EEG Epilepsy v3.0.0 corpus), then from the repo root:
#*   bash run_sweep.sh                            # full grid, foreground
#*   nohup bash run_sweep.sh > sweep.log 2>&1 &   # background for the ~2-3 h full grid
#* Override any setting inline, e.g.:  SEEDS="0 1 2" LR=5e-4 bash run_sweep.sh
#*
#* WARNING: regenerating the split overwrites <DATA_PATH>/TUEP_data/*.h5 each seed.
#*----------------------------------------------------------------------------*
set -euo pipefail

# ---- paths (EDIT ROOT_DIR; the other two default to the known HPC locations) ----
ROOT_DIR="${ROOT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/data/v3.0.0}"  # <-- EDIT: dir with 00_epilepsy/ 01_no_epilepsy/
export DATA_PATH="${DATA_PATH:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/BioFoundation}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/checkpoints}"

# ---- sweep settings (override via env if desired) ----
SEEDS="${SEEDS:-0 1 2 3 4}"
WINDOW_S="${WINDOW_S:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
RESULTS_CSV="${RESULTS_CSV:-sweep_results.csv}"

# Run from the repo root (where this script and sweep_foundation_models.py live).
cd "$(dirname "$0")"

[ -d "$ROOT_DIR" ] || { echo "ERROR: ROOT_DIR not found: $ROOT_DIR (edit it in run_sweep.sh)"; exit 1; }

echo "ROOT_DIR=$ROOT_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "seeds=[$SEEDS] window_s=$WINDOW_S batch=$BATCH_SIZE lr=$LR -> $RESULTS_CSV"

# Optional 1-cell smoke test (uncomment to validate the pipeline before the full grid):
# python -u sweep_foundation_models.py --root_dir "$ROOT_DIR" \
#     --seeds 0 --variants mixed_300 --modes frozen full \
#     --window_s "$WINDOW_S" --batch_size "$BATCH_SIZE" --lr "$LR" --results_csv "$RESULTS_CSV"

# Full grid: seeds x 4 variants x {frozen, full}. $SEEDS is intentionally unquoted (nargs).
python -u sweep_foundation_models.py --root_dir "$ROOT_DIR" \
    --seeds $SEEDS --window_s "$WINDOW_S" --batch_size "$BATCH_SIZE" --lr "$LR" \
    --results_csv "$RESULTS_CSV"

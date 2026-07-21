#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Inference-only score dump for the finished LuMamba sweep (for ROC curves).
#*
#* For each window length in {15,30,45,60}s and seed in {0..4}: regenerate the subject split
#* (deterministic from seed + window_s) and, for every variant, RELOAD the existing finetuned
#* checkpoint and dump its per-window/per-subject scores to $DUMP_DIR (no re-training). Reuses
#* the sweep's build + checkpoint-finding logic via --dump_only. Resumable: cells whose npz
#* already exist are skipped.
#*
#* PREREQUISITE: the sweep's checkpoints AND per-(ws,seed) manifests must still exist:
#*   $CHECKPOINT_DIR/checkpoints/sweep_w<ws>_s<seed>_<variant>_full/   (checkpoints)
#*   $MANIFEST_ROOT/w<ws>_s<seed>/windows_{test,val}.csv               (window record)
#*
#* EDIT ROOT_DIR, then from the repo root:
#*   nohup bash run_dump_predictions.sh > dump_predictions.log 2>&1 &
#* Then plot:  python scripts/plot_roc_variants.py --dump_dir <DUMP_DIR> --level subject
#*
#* The HDF5 is rebuilt by cropping the persisted manifest's EXACT windows (read-only), so the
#* manifests HYDRA reuses are NOT rewritten. Rebuilding still overwrites <DATA_PATH>/TUEP_data/*.h5
#* each build (scores persist in DUMP_DIR).
#*----------------------------------------------------------------------------*
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data/v3.0.0}"  # <-- EDIT: dir with 00_epilepsy/ 01_no_epilepsy/
export DATA_PATH="${DATA_PATH:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/BioFoundation}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/checkpoints}"

WINDOWS="${WINDOWS:-15 30 45 60}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SPLITS="${SPLITS:-test val}"
# Best models only (top-2 by subject/window AUROC). Override to add/replace, e.g.
# VARIANTS="lejepa_only_128 mixed_300 mixed_128" or all four.
VARIANTS="${VARIANTS:-lejepa_only_128 mixed_300}"
DUMP_DIR="${DUMP_DIR:-$DATA_PATH/roc_dumps}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$DATA_PATH/manifests}"  # must match the sweep's (run_lumamba_vs_hydra.sh)

cd "$(dirname "$0")"
[ -d "$ROOT_DIR" ] || { echo "ERROR: ROOT_DIR not found: $ROOT_DIR (edit run_dump_predictions.sh)"; exit 1; }

echo "ROOT_DIR=$ROOT_DIR | DATA_PATH=$DATA_PATH | CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "variants=[$VARIANTS] windows=[$WINDOWS] seeds=[$SEEDS] splits=[$SPLITS] -> dumps in $DUMP_DIR"

for WS in $WINDOWS; do
    # Inference batch can be larger than training; still scale down for long windows to respect
    # the CUDA grid cap (the eval also auto-shrinks). Override with BATCH_SIZE=... .
    if [ -n "${BATCH_SIZE:-}" ]; then BW="$BATCH_SIZE"; else BW=$(( 3840 / WS )); fi
    echo "==================== dump window_s=${WS}s  batch=${BW} ===================="
    python -u sweep_foundation_models.py --root_dir "$ROOT_DIR" \
        --window_s "$WS" --seeds $SEEDS \
        --variants $VARIANTS --modes full \
        --batch_size "$BW" --splits $SPLITS --manifest_root "$MANIFEST_ROOT" \
        --dump_only --dump_dir "$DUMP_DIR"
done

echo "Score dump done -> $DUMP_DIR"
echo "Plot ROC: python scripts/plot_roc_variants.py --dump_dir $DUMP_DIR --level subject --split test --variants $VARIANTS"
echo "Plot PR:  python scripts/plot_roc_variants.py --curve pr --sens_marker 0.85 --dump_dir $DUMP_DIR --level subject --split test --variants $VARIANTS"

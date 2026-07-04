#*----------------------------------------------------------------------------*
#* Subject-level evaluation for a finetuned LuMamba (or any FinetuneTask) checkpoint.
#*
#* The training test metrics are per WINDOW; the HYDRA baseline this project compares
#* against is per SUBJECT. This script closes that gap: it runs a finetuned checkpoint over
#* an HDF5 split, averages the per-window epilepsy probability within each subject, and
#* reports subject-level metrics (AUROC, balanced accuracy, sensitivity/specificity, ...)
#* alongside the window-level ones. It reuses the SAME Hydra config as run_train.py so the
#* model is instantiated identically, then loads the finetuned weights.
#*
#* Requires the HDF5 to carry a per-window 'subject' dataset (process_tuep_eeg.py +
#* make_hdf5.py write it); regenerate the split if an older HDF5 lacks it.
#*
#* Run on the HPC, from the repo root, composing the same experiment plus the checkpoint:
#*
#*   export DATA_PATH=/work/.../BioFoundation
#*   export CHECKPOINT_DIR=/work/.../checkpoints
#*   python -u eval_subject_level.py +experiment=LuMamba_finetune \
#*       +eval_checkpoint=/abs/path/to/best.ckpt [+eval_split=test]
#*----------------------------------------------------------------------------*

import os
from collections import defaultdict

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from datasets.tuh_dataset import CHN_ORDER
from models.modules.channel_embeddings import get_channel_locations

OmegaConf.register_new_resolver("env", lambda key: os.getenv(key), replace=True)
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method, replace=True)

# Same torch 2.6 weights_only workaround as run_train.py: our Lightning .ckpt embeds
# OmegaConf objects in hyper_parameters, which the weights_only=True default refuses.
_torch_load_orig = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_full


def _channelwise_normalize(x, eps=1e-8):
    """Per-channel z-score over time, matching tasks.finetune_task_LUNA.ChannelWiseNormalize."""
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    return (x - mean) / (std + eps)


def _metrics(y_true, prob):
    """Threshold-free (AUROC/AP) plus 0.5-threshold label metrics."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    both = len(np.unique(y_true)) > 1
    return {
        "auroc": roc_auc_score(y_true, prob) if both else float("nan"),
        "avg_precision": average_precision_score(y_true, prob) if both else float("nan"),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
    }


def _decode(s):
    return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)


@hydra.main(config_path="./config", config_name="defaults", version_base="1.1")
def run(cfg: DictConfig):
    # Lightning names checkpoints 'epoch=..-step=...ckpt'; the '=' breaks Hydra's override
    # parser, so also accept the path from the EVAL_CHECKPOINT env var (shell-safe, no quoting).
    ckpt_path = cfg.get("eval_checkpoint", None) or os.getenv("EVAL_CHECKPOINT")
    if not ckpt_path:
        raise SystemExit(
            "Pass the finetuned checkpoint via EVAL_CHECKPOINT=/abs/path (recommended: the '=' in "
            "Lightning ckpt names trips Hydra) or +eval_checkpoint='/abs/path/best.ckpt'."
        )
    split = cfg.get("eval_split", None) or os.getenv("EVAL_SPLIT", "test")
    hdf5_file = cfg.data_module[split].hdf5_file
    num_channels = int(cfg.data_module[split].get("num_channels", 20))
    batch = int(cfg.get("eval_batch_size", cfg.get("batch_size", 512)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the model exactly as run_train.py does, then load the finetuned weights.
    task = hydra.utils.instantiate(cfg.task, cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = task.load_state_dict(state, strict=False)
    print(f"===> Loaded {ckpt_path}")
    print(f"     missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"     (unexpected sample: {list(unexpected)[:4]})")
    model = task.model.to(device).eval()

    # Channel locations, identical to TUH_Dataset: (C, 3), batched to (B, C, 3) per forward.
    ch_loc = torch.tensor(
        np.stack(get_channel_locations(CHN_ORDER[:num_channels]), axis=0), dtype=torch.float32, device=device
    )

    probs, ys, subjects = [], [], []
    with h5py.File(hdf5_file, "r") as d:
        keys = list(d.keys())
        if keys and "subject" not in d[keys[0]]:
            raise SystemExit(
                f"{hdf5_file} has no per-window 'subject' dataset. Regenerate the split with the "
                "updated process_tuep_eeg.py + make_hdf5.py so subject ids are stored."
            )
        # The channel cross-attention runs a TransformerEncoder over (B*num_patches, C, E); its
        # fused CUDA kernel caps the grid at 65535, so B*num_patches must stay under that or the
        # forward dies with "CUDA error: invalid configuration argument". Shrink the batch for
        # long windows (num_patches = T // patch_size, patch_size=40) so eval works at any length.
        if keys:
            t_samples = int(d[keys[0]]["X"].shape[-1])
            num_patches = max(1, t_samples // 40)
            safe = max(1, min(batch, 60000 // num_patches))
            if safe < batch:
                print(f"     reducing eval batch {batch} -> {safe} ({num_patches} patches/window, CUDA grid cap)")
            batch = safe
        for k in keys:
            grp = d[k]
            X_all, y_all, subj_all = grp["X"], grp["y"][:], grp["subject"][:]
            n = X_all.shape[0]
            for i in range(0, n, batch):
                xb = torch.from_numpy(X_all[i:i + batch]).float().to(device)
                xb = _channelwise_normalize(xb)
                mask = torch.zeros(xb.shape[0], xb.shape[1], xb.shape[2], dtype=torch.bool, device=device)
                loc = ch_loc.unsqueeze(0).expand(xb.shape[0], -1, -1)
                with torch.no_grad():
                    logits, _ = model(xb, mask, loc)
                    p = torch.softmax(logits.float(), dim=1)[:, 1]  # P(epilepsy), class index 1
                probs.append(p.cpu().numpy())
                ys.append(np.asarray(y_all[i:i + batch]))
                subjects.append(subj_all[i:i + batch])

    prob = np.concatenate(probs)
    y = np.concatenate(ys).astype(int)
    subj = np.array([_decode(s) for s in np.concatenate(subjects)])

    win_m = _metrics(y, prob)
    print(f"\n=== {split.upper()} WINDOW-LEVEL (n={len(y)} windows) ===")
    for kk, vv in win_m.items():
        print(f"  {kk:14s} {vv:.4f}")
    # Machine-readable line for the sweep orchestrator (grep 'RESULT ... level=window').
    print("RESULT split={} level=window n={} {}".format(
        split, len(y), " ".join(f"{k}={v:.4f}" for k, v in win_m.items())))

    # Subject-level: mean epilepsy probability per subject; label is constant within subject.
    by_prob, by_lab = defaultdict(list), {}
    for p, yy, s in zip(prob, y, subj):
        by_prob[s].append(p)
        by_lab[s] = yy
    subj_ids = sorted(by_prob)
    s_prob = np.array([np.mean(by_prob[s]) for s in subj_ids])
    s_true = np.array([by_lab[s] for s in subj_ids])
    n_pos = int(s_true.sum())
    win_per_subj = np.array([len(by_prob[s]) for s in subj_ids])
    subj_m = _metrics(s_true, s_prob)
    print(f"\n=== {split.upper()} SUBJECT-LEVEL (n={len(subj_ids)} subjects: "
          f"{n_pos} epilepsy / {len(subj_ids) - n_pos} no-epilepsy; "
          f"{win_per_subj.min()}-{win_per_subj.max()} windows/subject, mean-prob aggregation) ===")
    for kk, vv in subj_m.items():
        print(f"  {kk:14s} {vv:.4f}")
    print("RESULT split={} level=subject n={} {}".format(
        split, len(subj_ids), " ".join(f"{k}={v:.4f}" for k, v in subj_m.items())))


if __name__ == "__main__":
    run()

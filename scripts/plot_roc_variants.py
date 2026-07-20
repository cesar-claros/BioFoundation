"""Plot ROC curves per LuMamba variant from the dumped eval scores.

Reads the per-cell score npz files written by ``eval_subject_level.py`` (``+dump_dir``),
i.e. ``sweep_foundation_models.py --dump_only --dump_dir <dir>``, named

    w<ws>_s<seed>_<variant>_<mode>_<split>.npz

each holding the per-window (``win_prob``/``win_y``/``win_subject``) and per-subject
(``subj_prob``/``subj_y``) scores. For each window length it overlays the variants' mean
ROC across seeds (with a +/-1 std band, computed by interpolating each seed's ROC onto a
common false-positive-rate grid) and reports the mean AUROC +/- std in the legend.

Run wherever the dump npz files and matplotlib/sklearn live (the HPC container).

    python scripts/plot_roc_variants.py --dump_dir <dir> --level subject --split test
    python scripts/plot_roc_variants.py --dump_dir <dir> --level window --window_s 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

# w<ws>_s<seed>_<variant>_<mode>_<split>.npz; variant may contain '_', so anchor mode/split.
FNAME = re.compile(r"^w(?P<ws>[0-9.]+)_s(?P<seed>\d+)_(?P<variant>.+)_(?P<mode>full|frozen)_(?P<split>\w+)\.npz$")
# The LuMamba pretraining variants plus 'hydra' (the same-windows HYDRA baseline, dumped by
# code/src/utils/trainer.py in the same npz schema), so all overlay in one ROC panel.
VARIANT_ORDER = ["reconstruction_only", "lejepa_only_128", "mixed_128", "mixed_300", "hydra"]
VARIANT_COLOR = {"reconstruction_only": "tab:gray", "lejepa_only_128": "tab:orange",
                 "mixed_128": "tab:green", "mixed_300": "tab:blue", "hydra": "tab:red"}
VARIANT_SHORT = {"reconstruction_only": "rec", "lejepa_only_128": "lejepa",
                 "mixed_128": "m128", "mixed_300": "m300", "hydra": "hydra"}
GRID = np.linspace(0.0, 1.0, 201)  # common FPR grid for vertical averaging


def _load(dump_dir: str, split: str, mode: str):
    """List (window_s, seed, variant, path) for the dumps matching split/mode."""
    recs = []
    for p in sorted(Path(dump_dir).glob(f"*_{split}.npz")):
        m = FNAME.match(p.name)
        if m and m["mode"] == mode and m["split"] == split:
            recs.append((float(m["ws"]), int(m["seed"]), m["variant"], p))
    return recs


def _roc(npz_path: Path, level: str):
    """(tpr-on-GRID, auroc) for one dump at the requested level, or None if single-class."""
    d = np.load(npz_path, allow_pickle=False)
    y = np.asarray(d["subj_y" if level == "subject" else "win_y"]).astype(int)
    s = np.asarray(d["subj_prob" if level == "subject" else "win_prob"], dtype=float)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    tpr_grid = np.interp(GRID, fpr, tpr)
    tpr_grid[0] = 0.0
    return tpr_grid, float(roc_auc_score(y, s))


def _panel(ax, recs_w, level, title, variants):
    """Draw the per-variant mean ROC (+/-1 std band) for one window on ``ax``."""
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--", zorder=1)  # chance
    for variant in variants:
        curves, aucs = [], []
        for _ws, _seed, var, path in recs_w:
            if var != variant:
                continue
            r = _roc(path, level)
            if r is not None:
                curves.append(r[0])
                aucs.append(r[1])
        if not curves:
            continue
        stack = np.vstack(curves)
        mean_tpr = stack.mean(0)
        std_tpr = stack.std(0)
        color = VARIANT_COLOR.get(variant, None)
        ax.plot(GRID, mean_tpr, color=color, lw=1.7, zorder=3,
                label=f"{variant}  AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} (n={len(aucs)})")
        ax.fill_between(GRID, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                        color=color, alpha=0.13, lw=0, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_dir", required=True, help="Directory of the *_<split>.npz score dumps.")
    p.add_argument("--level", default="subject", choices=["subject", "window"],
                   help="ROC on subject-aggregated scores (mean-prob per subject) or per window.")
    p.add_argument("--split", default="test", help="Which split's dumps to plot (default test).")
    p.add_argument("--mode", default="full", choices=["full", "frozen"], help="Finetune mode to plot.")
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANT_ORDER,
                   help="Restrict to these variants (e.g. the best models: lejepa_only_128 mixed_300); "
                   "default: every variant with dumps present.")
    p.add_argument("--window_s", type=float, default=None,
                   help="Restrict to one window length (Hz); default: one panel per window found.")
    p.add_argument("--out", default=None, help="Output PNG (default: <dump_dir>/roc_variants_<level>_<split>.png).")
    args = p.parse_args()

    recs = _load(args.dump_dir, args.split, args.mode)
    if not recs:
        raise SystemExit(f"No '*_{args.split}.npz' dumps ({args.mode} mode) in {args.dump_dir}; "
                         "run the sweep with --dump_only --dump_dir first.")
    # Variants to draw, in canonical order, restricted to --variants and to those actually dumped.
    present = {var for _ws, _seed, var, _p in recs}
    plot_variants = [v for v in VARIANT_ORDER if (args.variants is None or v in args.variants)]
    missing = [v for v in plot_variants if v not in present]
    if missing:
        print(f"note: no dumps for {missing} in {args.dump_dir}; skipping them.")
    plot_variants = [v for v in plot_variants if v in present]
    if not plot_variants:
        raise SystemExit(f"None of the requested variants have dumps in {args.dump_dir}.")
    windows = sorted({ws for ws, *_ in recs})
    if args.window_s is not None:
        windows = [w for w in windows if np.isclose(w, args.window_s)]
        if not windows:
            raise SystemExit(f"No dumps for window_s={args.window_s}; found {sorted({ws for ws, *_ in recs})}.")

    ncols = min(len(windows), 2)
    nrows = int(np.ceil(len(windows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, ws in enumerate(windows):
        recs_w = [r for r in recs if np.isclose(r[0], ws)]
        _panel(axes[i], recs_w, args.level, f"{ws:g}s windows", plot_variants)
    for ax in axes[len(windows):]:
        ax.axis("off")

    fig.suptitle(f"LuMamba ROC by variant ({args.level}-level, {args.split}; mean ±1 std over seeds)",
                 fontsize=12)
    fig.supxlabel("false positive rate (1 - specificity)")
    fig.supylabel("true positive rate (sensitivity)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    vtag = "" if args.variants is None else "_" + "-".join(VARIANT_SHORT[v] for v in plot_variants)
    suffix = f"_w{args.window_s:g}" if args.window_s is not None else ""
    out = (Path(args.out) if args.out else
           Path(args.dump_dir) / f"roc_variants_{args.level}_{args.split}{vtag}{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  ({len(windows)} window panel(s), variants={plot_variants})")


if __name__ == "__main__":
    main()

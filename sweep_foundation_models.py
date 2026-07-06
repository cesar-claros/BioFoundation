#*----------------------------------------------------------------------------*
#* Multi-seed sweep of the pretrained foundation models on TUEP epilepsy diagnosis.
#*
#* Grid: seeds x {frozen backbone, full finetune} x {all 4 LuMamba variants}. For each seed
#* it regenerates the subject-level split (process_tuep_eeg.py --seed), then for every
#* (variant, mode) it finetunes (run_train.py) and evaluates SUBJECT-level + window-level
#* metrics (eval_subject_level.py) on test and val. Results are appended to a CSV as they
#* complete, and a mean +/- std summary over seeds is printed at the end. This turns the
#* noisy single-split numbers into an estimate with error bars, so LuMamba variants can be
#* ranked against each other and against HYDRA's ~0.63 subject-level baseline.
#*
#* Checkpoint selection uses val_BinaryAUROC (max), not val_loss, since val_loss overfits
#* fast and selects undertrained epochs. Resumable: cells already in the CSV are skipped.
#*
#* Run on the HPC from the repo root, with DATA_PATH / CHECKPOINT_DIR exported (or passed):
#*
#*   export DATA_PATH=/work/.../BioFoundation
#*   export CHECKPOINT_DIR=/work/.../checkpoints
#*   nohup python -u sweep_foundation_models.py \
#*       --root_dir /path/to/tuh_eeg_epilepsy/v3.0.0 \
#*       --seeds 0 1 2 3 4 --window_s 30 --batch_size 64 --lr 1e-4 \
#*       > sweep.log 2>&1 &
#*
#* WARNING: regenerating the split overwrites <DATA_PATH>/TUEP_data/*.h5 each seed. The full
#* default grid is 5 seeds x 4 variants x 2 modes = 40 finetunes (~2-3 h at 30 s); subset
#* with --seeds / --variants / --modes, and interrupt/resume freely (CSV-backed).
#*----------------------------------------------------------------------------*

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ["reconstruction_only", "lejepa_only_128", "mixed_128", "mixed_300"]
MODES = {"frozen": "True", "full": "False"}  # mode name -> finetuning.freeze_layers value
METRIC_KEYS = ["auroc", "avg_precision", "balanced_acc", "accuracy",
               "sensitivity", "specificity", "f1", "cohen_kappa"]
CSV_FIELDS = ["window_s", "seed", "variant", "mode", "split", "level", "n", "threshold", *METRIC_KEYS]
RESULT_RE = re.compile(r"^RESULT split=(\S+) level=(\S+) n=(\d+) (.+)$")


def sh(cmd, env=None, capture=False):
    """Run a subprocess; stream output unless capturing. Raise on non-zero exit."""
    print("  $ " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, env=env, capture_output=capture, text=True)
    if res.returncode != 0:
        if capture:
            sys.stdout.write(res.stdout or "")
            sys.stderr.write(res.stderr or "")
        raise RuntimeError(f"command failed (exit {res.returncode}): {' '.join(cmd)}")
    return res.stdout if capture else ""


def parse_results(stdout):
    """{(split, level): {n, metric: value, ...}} from the eval script's RESULT lines."""
    out = {}
    for line in stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if not m:
            continue
        split, level, n, kvs = m.groups()
        d = {"n": int(n)}
        for pair in kvs.split():
            k, v = pair.split("=")
            d[k] = float(v)
        out[(split, level)] = d
    return out


def find_best_ckpt(checkpoint_dir, tag):
    """Best (monitored) checkpoint for a run tag, else last.ckpt, else None."""
    base = Path(checkpoint_dir) / "checkpoints" / tag
    for pattern in ("*/epoch=*.ckpt", "*/last.ckpt"):
        cands = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
        if cands:
            return str(cands[-1])
    return None


def summarize(rows):
    # Report subject-level AUROC (threshold-free) and balanced accuracy at the val-calibrated
    # threshold (level=subject_cal); fall back to the 0.5 threshold (level=subject) if calibration
    # was disabled. Grouped by (window_s, variant, mode, split).
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["level"] not in ("subject", "subject_cal"):
            continue
        key = (r["window_s"], r["variant"], r["mode"], r["split"])
        try:
            if r["level"] == "subject":
                agg[key]["auroc"].append(float(r["auroc"]))
                agg[key].setdefault("balanced_acc", [])  # ensure the key exists
            if r["level"] == "subject_cal":
                agg[key]["balanced_acc_cal"].append(float(r["balanced_acc"]))
        except (ValueError, TypeError, KeyError):
            pass

    def ms(xs):
        if not xs:
            return "n/a"
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return f"{statistics.mean(xs):.3f}+/-{sd:.3f}"

    print("\n===== SUBJECT-LEVEL SUMMARY (mean +/- std over seeds; bal_acc at val-calibrated thr) =====")
    print(f"{'win_s':6s} {'variant':20s} {'mode':6s} {'split':5s} {'seeds':5s} {'AUROC':16s} {'bal_acc_cal':16s}")
    for key in sorted(agg):
        window_s, variant, mode, split = key
        a = agg[key]["auroc"]
        b = agg[key].get("balanced_acc_cal", [])
        print(f"{str(window_s):6s} {variant:20s} {mode:6s} {split:5s} {len(a):<5d} {ms(a):16s} {ms(b):16s}")
    print("Compare against HYDRA (same windows, same calibration) subject-level balanced accuracy.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root_dir", required=True, help="TUH EEG Epilepsy v3.0.0 dir (for regeneration).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--window_s", type=float, default=30.0)
    p.add_argument("--min_duration_s", type=float, default=120.0)
    p.add_argument("--max_windows_per_subject", type=int, default=200)
    p.add_argument("--processes", type=int, default=24, help="Workers for process_tuep_eeg.")
    p.add_argument("--batch_size", type=int, default=64, help="Finetune + eval batch size.")
    p.add_argument("--lr", type=float, default=None, help="optimizer.lr override (default: config's 5e-4).")
    p.add_argument("--max_epochs", type=int, default=None, help="trainer.max_epochs override (default: config's 50).")
    p.add_argument("--monitor", default="val_BinaryAUROC", help="Checkpoint/early-stop monitor metric.")
    p.add_argument("--monitor_mode", default="max")
    p.add_argument("--splits", nargs="+", default=["test", "val"])
    p.add_argument("--calib_split", default="val",
                   help="Held-out split for subject-level threshold calibration (applied when scoring test); "
                   "'' disables. The eval reports subject_cal metrics at the calibrated threshold.")
    p.add_argument("--manifest_root", default=None,
                   help="Persist each build's window manifests to <manifest_root>/w<window_s>_s<seed>/ so HYDRA "
                   "can reuse the exact same windows (default: <output_dir>/manifests).")
    p.add_argument("--results_csv", default="sweep_results.csv")
    p.add_argument("--output_dir", default=os.getenv("DATA_PATH"),
                   help="Where TUEP_data/ is written (default $DATA_PATH).")
    p.add_argument("--checkpoint_dir", default=os.getenv("CHECKPOINT_DIR"),
                   help="Run outputs root (default $CHECKPOINT_DIR).")
    p.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    args = p.parse_args()

    if not args.output_dir or not args.checkpoint_dir:
        raise SystemExit("Set DATA_PATH and CHECKPOINT_DIR (env) or pass --output_dir / --checkpoint_dir.")
    env = os.environ.copy()
    env["DATA_PATH"] = args.output_dir
    env["CHECKPOINT_DIR"] = args.checkpoint_dir
    py = sys.executable
    ws = float(args.window_s)
    ws_tag = f"{args.window_s:g}"  # clean tag/dir label, e.g. 30 not 30.0
    manifest_root = args.manifest_root or os.path.join(args.output_dir, "manifests")

    # Resume: load any prior results and skip completed (seed, variant, mode, split, level) cells.
    rows, done = [], set()
    csv_path = Path(args.results_csv)
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((float(r["window_s"]), int(r["seed"]), r["variant"], r["mode"], r["split"], r["level"]))
        print(f"Resuming from {csv_path}: {len(done)} result cells already present.")

    def write_csv():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

    for seed in args.seeds:
        manifest_dir = os.path.join(manifest_root, f"w{ws_tag}_s{seed}")
        need_regen = any(
            (ws, seed, v, m, sp, "subject") not in done
            for v in args.variants for m in args.modes for sp in args.splits
        )
        print(f"\n########## {ws_tag}s windows | SEED {seed} ##########", flush=True)
        if need_regen:
            regen = [py, "make_datasets/process_tuep_eeg.py",
                     "--root_dir", args.root_dir, "--output_dir", args.output_dir, "--interictal",
                     "--seed", str(seed), "--window_s", str(args.window_s),
                     "--min_duration_s", str(args.min_duration_s),
                     "--max_windows_per_subject", str(args.max_windows_per_subject),
                     "--manifest_dir", manifest_dir, "--processes", str(args.processes)]
            if args.dry_run:
                print("  $ " + " ".join(regen))
            else:
                sh(regen, env=env)
        else:
            print("  all cells done for this build; skipping regeneration.")

        for variant in args.variants:
            for mode in args.modes:
                pending = [sp for sp in args.splits if (ws, seed, variant, mode, sp, "subject") not in done]
                if not pending:
                    print(f"  [skip] w{ws_tag} seed={seed} {variant}/{mode} already done", flush=True)
                    continue
                tag = f"sweep_w{ws_tag}_s{seed}_{variant}_{mode}"
                print(f"\n---- seed={seed} variant={variant} mode={mode} (tag={tag}) ----", flush=True)
                ft = [py, "-u", "run_train.py", "+experiment=LuMamba_finetune",
                      f"pretrained_variant={variant}", f"finetuning.freeze_layers={MODES[mode]}",
                      f"batch_size={args.batch_size}", f"seed={seed}", f"tag={tag}",
                      f"model_checkpoint.monitor={args.monitor}", f"model_checkpoint.mode={args.monitor_mode}",
                      f"callbacks.early_stopping.monitor={args.monitor}",
                      f"callbacks.early_stopping.mode={args.monitor_mode}"]
                # Frozen backbone leaves most params without grad, so DDP must allow unused params.
                if mode == "frozen":
                    ft.append("find_unused_parameters=true")
                if args.lr is not None:
                    ft.append(f"optimizer.lr={args.lr}")
                if args.max_epochs is not None:
                    ft.append(f"trainer.max_epochs={args.max_epochs}")

                if args.dry_run:
                    print("  $ " + " ".join(ft))
                    for sp in pending:
                        cal = f" +calib_split={args.calib_split}" if args.calib_split and args.calib_split != sp else ""
                        print(f"  $ EVAL_CHECKPOINT=<best> {py} -u eval_subject_level.py "
                              f"+experiment=LuMamba_finetune +eval_split={sp} +eval_batch_size={args.batch_size}{cal}")
                    continue

                try:
                    sh(ft, env=env)
                except RuntimeError as e:
                    print(f"  !! finetune failed, skipping: {e}", flush=True)
                    continue
                ckpt = find_best_ckpt(args.checkpoint_dir, tag)
                if ckpt is None:
                    print(f"  !! no checkpoint found for {tag}; skipping eval", flush=True)
                    continue
                print(f"  best checkpoint: {ckpt}", flush=True)

                for sp in pending:
                    ev = [py, "-u", "eval_subject_level.py", "+experiment=LuMamba_finetune",
                          f"+eval_split={sp}", f"+eval_batch_size={args.batch_size}"]
                    # Calibrate the subject threshold on --calib_split when scoring a different split.
                    if args.calib_split and args.calib_split != sp:
                        ev.append(f"+calib_split={args.calib_split}")
                    try:
                        out = sh(ev, env=dict(env, EVAL_CHECKPOINT=ckpt), capture=True)
                    except RuntimeError as e:
                        print(f"  !! eval ({sp}) failed, skipping: {e}", flush=True)
                        continue
                    print(out)
                    for (rsplit, level), m in parse_results(out).items():
                        rows.append({"window_s": ws, "seed": seed, "variant": variant, "mode": mode,
                                     "split": rsplit, "level": level, "n": m["n"],
                                     "threshold": m.get("threshold", ""),
                                     **{k: m.get(k, "") for k in METRIC_KEYS}})
                        done.add((ws, seed, variant, mode, rsplit, level))
                    write_csv()

    if not args.dry_run:
        summarize(rows)
        print(f"\nWrote {csv_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()

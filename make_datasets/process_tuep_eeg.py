#*----------------------------------------------------------------------------*
#* TUH EEG Epilepsy (TUEP, v3.0.0) preprocessing for the LuMamba pretrained model.
#*
#* Mirrors process_raw_eeg.py (tuab / file-level binary label) but adapted to the TUH
#* EEG Epilepsy corpus and to feed the LuMamba / LUNA finetune dataset (datasets.tuh_dataset.
#* TUH_Dataset). Per recording: read EDF -> rename LE->REF -> band-pass 0.1-75 Hz -> notch
#* 60 Hz -> resample 256 Hz -> TCP bipolar (the 20 NON-ear pairs, in CHN_ORDER order) ->
#* fixed-length windows (--window_s, default 5 s -> (20, 1280)) -> one {"X", "y", "subject"}
#* pickle per window. Label is per-patient
#* diagnosis: epilepsy (00_epilepsy) = 1, no-epilepsy (01_no_epilepsy) = 0. Then the pickles
#* are bundled into TUEP_data/{train,val,test}.h5 with the shared make_hdf5.create_hdf5.
#*
#* Why the 20 NON-ear pairs (drop A1-T3, T4-A2): TUH_Dataset.CHN_ORDER lists the ear pairs
#* LAST, so a 20-channel array is exactly CHN_ORDER[:20]; the '_a' montages lack the ear
#* channels, and mixing 20/22-channel arrays would break the HDF5 bundling. 20 channels is
#* also the intersection this project already uses for its HYDRA bipolar montage.
#*
#*   python make_datasets/process_tuep_eeg.py \
#*       --root_dir /path/to/tuh_eeg_epilepsy/v3.0.0 \
#*       --output_dir /path/to/processed_eeg [--interictal] [--processes 24]
#*
#* Balance the splits by capping windows PER SUBJECT (so a few patients with many
#* recordings do not dominate; the budget is spread evenly across each subject's
#* recordings), and optionally reuse a HYDRA run's subject-level split so the held-out
#* subjects match for an apples-to-apples LuMamba-vs-HYDRA comparison:
#*   python make_datasets/process_tuep_eeg.py --root_dir ... --output_dir ... \
#*       --interictal --min_duration_s 120 --max_windows_per_subject 200 \
#*       --hydra_windows_dir /path/to/tuh-eeg-epilepsy/logs/train/runs/<timestamp>
#*
#* Then point config/experiment/LuMamba_finetune.yaml at:
#*   data_module.{train,val,test}._target_: datasets.tuh_dataset.TUH_Dataset
#*   data_module.{train,val,test}.hdf5_file:  <output_dir>/TUEP_data/{train,val,test}.h5
#*   data_module.{train,val,test}.num_channels: 20   (CHN_ORDER[:20])
#*----------------------------------------------------------------------------*

import argparse
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hdf5 import create_hdf5  # noqa: E402  (sibling script, bundles pkls -> .h5)

# --- The 20 non-ear TCP bipolar pairs, in TUH_Dataset.CHN_ORDER order (ears dropped). ---
# Each value is (bipolar name, anode electrode, cathode electrode) using the BARE electrode
# name (e.g. "FP1", "T3"). Channels are matched by electrode via _electrode() below, which
# drops the "EEG " prefix and the "-REF"/"-LE" reference suffix -- robust to however MNE
# surfaces the TUH labels, and montage-agnostic (AR/LE and their _a variants).
BIPOLAR_PAIRS = [
    ("FP1-F7", "FP1", "F7"), ("F7-T3", "F7", "T3"),
    ("T3-T5", "T3", "T5"),   ("T5-O1", "T5", "O1"),
    ("FP2-F8", "FP2", "F8"), ("F8-T4", "F8", "T4"),
    ("T4-T6", "T4", "T6"),   ("T6-O2", "T6", "O2"),
    ("T3-C3", "T3", "C3"),   ("C3-CZ", "C3", "CZ"),
    ("CZ-C4", "CZ", "C4"),   ("C4-T4", "C4", "T4"),
    ("FP1-F3", "FP1", "F3"), ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),   ("P3-O1", "P3", "O1"),
    ("FP2-F4", "FP2", "F4"), ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),   ("P4-O2", "P4", "O2"),
]
N_BIPOLAR = len(BIPOLAR_PAIRS)  # 20
SFREQ = 256
DEFAULT_WINDOW_S = 5.0          # 5 s at 256 Hz -> 1280 samples (LuMamba's pretraining window)
COHORT_LABEL = {"00_epilepsy": 1, "01_no_epilepsy": 0}


def _electrode(ch_name: str) -> str:
    """Bare electrode name from a raw EDF label (mirrors the engine's _rename_channels):
    drop the 'EEG ' prefix and the '-REF'/'-LE' suffix, case/space-insensitive.
    'EEG FP1-REF' / 'FP1-REF' / 'Fp1' -> 'FP1'."""
    return ch_name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()


def make_bipolar_20(raw):
    """(20, T) float array in CHN_ORDER[:20] order, or None if any of the 20 pairs is
    underivable (a required electrode is absent)."""
    data = raw.get_data(units="uV")
    idx = {_electrode(n): i for i, n in enumerate(raw.ch_names)}
    out = []
    for _name, e1, e2 in BIPOLAR_PAIRS:
        if e1 not in idx or e2 not in idx:
            return None
        out.append(data[idx[e1]] - data[idx[e2]])
    return np.asarray(out, dtype=np.float32)


def _preprocess_bipolar(file_path, min_duration_s):
    """Read one EDF and return its (20, T) TCP-bipolar array at SFREQ (band-pass 0.1-75 Hz, notch
    60 Hz, resample 256 Hz), or None if the recording is shorter than ``min_duration_s``. Raises
    ValueError if the 20 bipolar pairs cannot be built. Shared by the normal build and the
    from-manifest rebuild so both crop byte-identical windows."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    # Keep the referential channels the bipolar montage needs, matched by electrode name (so the
    # exact TUH label format / EEG prefix does not matter). '_a' montages just lack the ear
    # channels, which we do not use.
    needed = {e for _n, a, b in BIPOLAR_PAIRS for e in (a, b)}
    present = [c for c in raw.ch_names if _electrode(c) in needed]
    found = {_electrode(c) for c in present}
    if len(found) < len(needed):
        raise ValueError(f"missing referential channels ({len(found)}/{len(needed)}); "
                         f"channels seen: {raw.ch_names[:25]}")
    raw.pick(present)
    # Drop recordings shorter than min_duration_s (native rate, before filtering): the 0.1 Hz
    # high-pass needs a ~68 s FIR filter, so shorter signals get edge distortion (RuntimeWarning
    # "filter_length ... longer than the signal").
    if min_duration_s and raw.n_times / float(raw.info["sfreq"]) < min_duration_s:
        return None
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
    raw.notch_filter(60, verbose=False)
    if int(round(raw.info["sfreq"])) != SFREQ:
        raw.resample(SFREQ, npad="auto", n_jobs=1, verbose=False)
    data = make_bipolar_20(raw)
    if data is None or data.shape[0] != N_BIPOLAR:
        raise ValueError("could not build the 20 bipolar channels")
    return data


def process_and_dump_file(params):
    """Worker: preprocess one EDF, dump its first ``max_windows`` fixed-length windows as pickles,
    and return a manifest of the windows kept (one dict per window) so the exact same time-segments
    can be reused by another model (e.g. HYDRA reads path/start/end and crops the native EDF)."""
    file_path, dump_folder, label, subject, max_windows, min_duration_s, window_samples = params
    stem = os.path.basename(file_path).split(".")[0]
    split = os.path.basename(dump_folder)
    window_s = window_samples / float(SFREQ)
    try:
        data = _preprocess_bipolar(file_path, min_duration_s)
        if data is None:
            return []
        n_times = data.shape[1]
        n_win = n_times // window_samples
        if max_windows is not None:
            n_win = min(n_win, max_windows)
        rows = []
        for i in range(n_win):
            seg = data[:, i * window_samples:(i + 1) * window_samples]
            with open(os.path.join(dump_folder, f"{stem}_{i}.pkl"), "wb") as f:
                pickle.dump({"X": seg, "y": int(label), "subject": subject}, f)
            # Window i spans [i*window_s, (i+1)*window_s] seconds of the recording. Resampling
            # preserves the time axis, so these bounds are valid on the native EDF too.
            rows.append({"subject": subject, "path": file_path, "split": split,
                         "label": int(label), "window_idx": i,
                         "start": round(i * window_s, 6), "end": round((i + 1) * window_s, 6),
                         "window_s": window_s, "sfreq": SFREQ})
        return rows
    except Exception as e:  # noqa: BLE001
        with open("tuep-process-errors.txt", "a") as f:
            f.write(f"Error processing {file_path}: {e}\n")
        return []


def dump_windows_from_manifest(params):
    """Worker for the from-manifest rebuild: crop EXACTLY the manifest-listed windows of one
    recording (by window_idx) and dump them as pickles, reusing the same preprocessing. No split /
    cap derivation and no manifest is written, so the persisted manifest stays the single source of
    truth. params: (file_path, dump_folder, label, subject, keep_indices, window_samples). Returns
    the number of windows dumped."""
    file_path, dump_folder, label, subject, keep_indices, window_samples = params
    stem = os.path.basename(file_path).split(".")[0]
    try:
        # The min_duration drop already happened at build time (short recordings are absent from
        # the manifest), so do not re-apply it here.
        data = _preprocess_bipolar(file_path, 0.0)
        if data is None:
            return 0
        n_times = data.shape[1]
        dumped = 0
        for i in sorted({int(k) for k in keep_indices}):
            a, b = i * window_samples, (i + 1) * window_samples
            if b > n_times:
                with open("tuep-process-errors.txt", "a") as f:
                    f.write(f"manifest window {i} out of range for {file_path} "
                            f"(need {b} <= {n_times}); preprocessing drifted, skipping\n")
                continue
            seg = data[:, a:b]
            with open(os.path.join(dump_folder, f"{stem}_{i}.pkl"), "wb") as f:
                pickle.dump({"X": seg, "y": int(label), "subject": subject}, f)
            dumped += 1
        return dumped
    except Exception as e:  # noqa: BLE001
        with open("tuep-process-errors.txt", "a") as f:
            f.write(f"Error rebuilding {file_path} from manifest: {e}\n")
        return 0


def _has_seizure(edf_path: Path) -> bool:
    """True if the sibling .csv_bi term annotation contains a 'seiz' event."""
    csv_bi = edf_path.with_suffix(".csv_bi")
    if not csv_bi.exists():
        return False
    try:
        df = pd.read_csv(csv_bi, comment="#")
        return bool((df.get("label", pd.Series(dtype=str)).astype(str) == "seiz").any())
    except Exception:  # noqa: BLE001
        return False


def collect_recordings(root_dir: Path, interictal: bool):
    """[(edf_path, subject, label)] over both cohorts, optionally interictal-only."""
    recs = []
    for cohort, label in COHORT_LABEL.items():
        cohort_dir = root_dir / cohort
        if not cohort_dir.is_dir():
            print(f"!! cohort dir not found: {cohort_dir}")
            continue
        for edf in sorted(cohort_dir.rglob("*.edf")):
            # Only raw recordings: <subject>_s<session>_t<recording>.edf. Skip derived EDFs
            # such as the HYDRA pipeline's <...>_ica.edf (IC-source channels, not electrodes).
            last = edf.stem.rsplit("_", 1)[-1]
            if not (last.startswith("t") and last[1:].isdigit()):
                continue
            subject = edf.relative_to(cohort_dir).parts[0]
            if interictal and _has_seizure(edf):
                continue
            recs.append((edf, subject, label))
    return recs


def split_subjects(recs, seed, ratios=(0.6, 0.2, 0.2)):
    """Subject-level split, stratified by class. Returns {split: set(subjects)}."""
    rng = np.random.RandomState(seed)
    subj_label = {}
    for _edf, subject, label in recs:
        subj_label[subject] = label
    splits = {"train": set(), "val": set(), "test": set()}
    for cls in (0, 1):
        subs = sorted(s for s, y in subj_label.items() if y == cls)
        rng.shuffle(subs)
        n = len(subs)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        splits["train"].update(subs[:n_tr])
        splits["val"].update(subs[n_tr:n_tr + n_va])
        splits["test"].update(subs[n_tr + n_va:])
    return splits


def load_hydra_split(windows_dir):
    """subject -> split from a HYDRA run's windows_{train,val,test}.csv 'subject' column.

    Reuses the exact subject-level split of a finished HYDRA run so LuMamba is trained
    and evaluated on the same held-out patients (apples-to-apples). The window CSVs list
    one row per window; we only need the unique subject IDs per split, which are the
    corpus subject-folder names, matching collect_recordings' ``subject``.
    """
    windows_dir = Path(windows_dir)
    subj_to_split = {}
    for split in ("train", "val", "test"):
        csv_path = windows_dir / f"windows_{split}.csv"
        if not csv_path.exists():
            raise SystemExit(f"HYDRA split CSV not found: {csv_path}")
        subs = pd.read_csv(csv_path, usecols=["subject"])["subject"].astype(str).unique()
        for s in subs:
            prev = subj_to_split.get(s)
            if prev is not None and prev != split:
                print(f"!! subject {s} appears in both '{prev}' and '{split}'; keeping '{prev}'.")
                continue
            subj_to_split[s] = split
    return subj_to_split


def per_recording_caps(recs, max_per_subject, max_per_recording):
    """Map each recording's EDF path to its window cap (int, or None = unlimited).

    A per-SUBJECT budget (``max_per_subject``) is distributed as evenly as possible over
    that subject's recordings: ``base = budget // n_rec`` windows each, and the first
    ``budget % n_rec`` recordings (ordered by path, so the choice is deterministic) get
    one extra. The per-recording caps therefore sum to exactly the budget, so a subject
    with many recordings no longer floods the pool (each of its recordings contributes
    only ``~budget / n_rec`` windows, possibly 0). A per-recording ceiling
    (``max_per_recording``) is then applied on top, the smaller of the two winning. With
    neither set, the cap is None (all windows are kept).
    """
    by_subject = defaultdict(list)
    for edf, subject, _label in recs:
        by_subject[subject].append(str(edf))
    caps = {}
    for paths in by_subject.values():
        paths = sorted(paths)
        n = len(paths)
        if max_per_subject is not None:
            base, extra = divmod(max_per_subject, n)
            sub_caps = [base + (1 if i < extra else 0) for i in range(n)]
        else:
            sub_caps = [None] * n
        for path, sub_cap in zip(paths, sub_caps):
            if sub_cap is None:
                caps[path] = max_per_recording
            elif max_per_recording is None:
                caps[path] = sub_cap
            else:
                caps[path] = min(sub_cap, max_per_recording)
    return caps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", required=True, help="Path to the TUH EEG Epilepsy v3.0.0 dir (has 00_epilepsy / 01_no_epilepsy).")
    parser.add_argument("--output_dir", required=True, help="Where TUEP_data/{processed,*.h5} is written.")
    parser.add_argument("--processes", type=int, default=24, help="Parallel worker processes (default 24).")
    parser.add_argument("--interictal", action="store_true", help="Drop recordings with a seizure annotation (.csv_bi 'seiz'); diagnosis task.")
    parser.add_argument("--max_windows_per_recording", type=int, default=None, help="Cap 5 s windows per recording (default: all).")
    parser.add_argument("--max_windows_per_subject", type=int, default=None,
                        help="Cap total 5 s windows per SUBJECT, spread evenly across that subject's recordings "
                        "(mirrors the HYDRA max_windows_per_subject). Keeps a handful of patients with many "
                        "recordings from dominating, so the window counts track the 60/20/20 subject split "
                        "instead of ballooning wherever the high-recording subjects land. Combined with "
                        "--max_windows_per_recording via the smaller cap.")
    parser.add_argument("--hydra_windows_dir", default=None,
                        help="Directory of a finished HYDRA run holding windows_{train,val,test}.csv. If given, "
                        "reuse that run's subject-level split (identical held-out patients for an apples-to-apples "
                        "LuMamba-vs-HYDRA comparison) instead of the seed split; pool subjects absent from those "
                        "CSVs are added to train, and HYDRA subjects absent from this pool are reported.")
    parser.add_argument("--min_duration_s", type=float, default=0.0,
                        help="Drop recordings shorter than this many seconds (0 = keep all). Use ~70+ to avoid the "
                        "0.1 Hz filter distorting short recordings; e.g. 120 for >= 2 min.")
    parser.add_argument("--seed", type=int, default=42, help="Subject-split seed (default 42).")
    parser.add_argument("--window_s", type=float, default=DEFAULT_WINDOW_S,
                        help="Window length in seconds (default 5 = LuMamba's pretraining window). Longer windows "
                        "give the model more temporal context (HYDRA uses 30-120 s); LuMamba is Mamba-based and "
                        "length-flexible, tokenizing T//40 samples into patches. Use a multiple of 5 s so T stays "
                        "divisible by the patch size (5 s=32 patches, 30 s=192, 60 s=384). Set --min_duration_s "
                        ">= this. Longer windows need a smaller finetune batch_size (more patches = more memory).")
    parser.add_argument("--keep_pkl", action="store_true", help="Keep the intermediate .pkl files after building the .h5 files.")
    parser.add_argument("--from_manifest", default=None,
                        help="Rebuild the HDF5 by cropping EXACTLY the windows recorded in a persisted manifest "
                        "directory (its windows_{train,val,test}.csv), instead of re-deriving the subject split "
                        "and re-windowing. Reads the manifest (does NOT rewrite it) so it stays the source of "
                        "truth, and guarantees the same windows the original build produced. Ignores --seed / "
                        "--interictal / --max_windows_per_subject (those are baked into the manifest). Used by the "
                        "inference-only score dump (sweep_foundation_models.py --dump_only).")
    parser.add_argument("--manifest_splits", nargs="+", default=["train", "val", "test"],
                        help="With --from_manifest, only rebuild these splits' HDF5 (e.g. 'test val' to skip the "
                        "large train build when only dumping test/val scores). Default: all three.")
    parser.add_argument("--manifest_dir", default=None,
                        help="Where to write windows_{train,val,test}.csv (the kept window time-segments for a "
                        "same-windows HYDRA comparison). Default: alongside the .h5 files. The sweep persists one "
                        "set per (window_s, seed).")
    args = parser.parse_args()
    window_samples = int(round(args.window_s * SFREQ))
    if args.min_duration_s and args.min_duration_s < args.window_s:
        print(f"!! --min_duration_s ({args.min_duration_s}) < --window_s ({args.window_s}); "
              "recordings shorter than one window yield 0 windows.")

    root_dir = Path(args.root_dir)
    base = os.path.join(args.output_dir, "TUEP_data")
    proc = os.path.join(base, "processed")
    # Start clean: drop any pkls left by a previous (interrupted or differently-configured)
    # run so stale/partial windows are never bundled into the HDF5s.
    shutil.rmtree(proc, ignore_errors=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(proc, split), exist_ok=True)

    # From-manifest rebuild (used by the inference-only score dump): reconstruct the HDF5 by
    # cropping EXACTLY the windows a persisted manifest records, WITHOUT re-deriving the split /
    # caps and WITHOUT rewriting the manifest. This keeps the manifest the single source of truth
    # (the same-windows HYDRA comparison relies on it) and guarantees the rebuilt windows match the
    # ones the checkpoint was trained / evaluated on, robust to any later drift in the split logic.
    if args.from_manifest:
        man_dir = Path(args.from_manifest)
        want = set(args.manifest_splits)
        frames = []
        for split in ("train", "val", "test"):
            csv_path = man_dir / f"windows_{split}.csv"
            if split in want and csv_path.exists():
                frames.append(pd.read_csv(csv_path))
        if not frames:
            raise SystemExit(f"No windows_{{{','.join(sorted(want))}}}.csv found in {man_dir}.")
        man = pd.concat(frames, ignore_index=True)
        man_ws = float(man["window_s"].iloc[0])
        if not np.isclose(man_ws, args.window_s):
            print(f"!! manifest window_s={man_ws} != --window_s={args.window_s}; using the manifest's.")
        ws_samples = int(round(man_ws * SFREQ))
        params = [
            (str(path), os.path.join(proc, split), int(label), str(subject),
             g["window_idx"].tolist(), ws_samples)
            for (path, split, subject, label), g in man.groupby(["path", "split", "subject", "label"])
        ]
        print(f"Rebuilding {len(params)} recordings ({len(man)} windows, splits={sorted(want)}) from "
              f"manifest {man_dir}; NOT rewriting it.")
        with Pool(processes=args.processes) as pool:
            counts = list(tqdm.tqdm(pool.imap_unordered(dump_windows_from_manifest, params),
                                    total=len(params)))
        print(f"Dumped {sum(counts)} windows across {len(params)} recordings.")
        for split in ("train", "val", "test"):
            if split in want:
                create_hdf5(os.path.join(proc, split), os.path.join(base, f"{split}.h5"), finetune=True)
        if not args.keep_pkl:
            shutil.rmtree(proc, ignore_errors=True)
        print(f"Done (from manifest). HDF5 at {base}/{{{','.join(sorted(want))}}}.h5")
        return

    recs = collect_recordings(root_dir, args.interictal)
    if not recs:
        raise SystemExit(f"No EDF recordings found under {root_dir}.")

    pool_subjects = {subject for _edf, subject, _label in recs}
    if args.hydra_windows_dir:
        hydra_map = load_hydra_split(args.hydra_windows_dir)
        subj_to_split = {}
        leftovers = []
        for s in sorted(pool_subjects):
            if s in hydra_map:
                subj_to_split[s] = hydra_map[s]
            else:
                subj_to_split[s] = "train"
                leftovers.append(s)
        missing = Counter(sp for s, sp in hydra_map.items() if s not in pool_subjects)
        print(f"Aligned split to HYDRA windows in {args.hydra_windows_dir}: "
              f"{len(hydra_map)} mapped subjects, {len(leftovers)} pool subjects not in HYDRA -> train, "
              f"{sum(missing.values())} HYDRA subjects absent from this pool {dict(missing)}.")
    else:
        splits = split_subjects(recs, args.seed)
        subj_to_split = {s: sp for sp, subs in splits.items() for s in subs}
    split_subj_counts = Counter(subj_to_split.values())
    print(f"Recordings: {len(recs)} | subjects: {len(subj_to_split)} "
          f"(train {split_subj_counts['train']}, val {split_subj_counts['val']}, test {split_subj_counts['test']})")

    caps = per_recording_caps(recs, args.max_windows_per_subject, args.max_windows_per_recording)
    params = [
        (str(edf), os.path.join(proc, subj_to_split[subject]), label, subject,
         caps[str(edf)], args.min_duration_s, window_samples)
        for edf, subject, label in recs
    ]
    print(f"Processing {len(params)} recordings with {args.processes} processes...")
    manifest_rows = []
    with Pool(processes=args.processes) as pool:
        for rows in tqdm.tqdm(pool.imap_unordered(process_and_dump_file, params), total=len(params)):
            if rows:
                manifest_rows.extend(rows)

    # Per-split window manifests: the exact (path, start, end) time-segments kept, so another
    # model (HYDRA) can crop the identical windows from the native EDFs for a same-windows
    # comparison. Written next to the HDF5s; --manifest_dir relocates them (the sweep persists
    # one set per (window_s, seed)).
    manifest_dir = args.manifest_dir or base
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_cols = ["subject", "path", "split", "label", "window_idx", "start", "end", "window_s", "sfreq"]
    for split in ("train", "val", "test"):
        split_rows = [r for r in manifest_rows if r["split"] == split]
        out_csv = os.path.join(manifest_dir, f"windows_{split}.csv")
        pd.DataFrame(split_rows, columns=manifest_cols).to_csv(out_csv, index=False)
    print(f"Wrote window manifests to {manifest_dir}/windows_{{train,val,test}}.csv "
          f"({len(manifest_rows)} windows total).")

    for split in ("train", "val", "test"):
        src = os.path.join(proc, split)
        tgt = os.path.join(base, f"{split}.h5")
        create_hdf5(src, tgt, finetune=True)
    if not args.keep_pkl:
        shutil.rmtree(proc, ignore_errors=True)
    print(f"Done. HDF5 at {base}/{{train,val,test}}.h5 "
          f"(X: (N, {N_BIPOLAR}, {window_samples}) = {args.window_s:g}s windows, y: 0/1). "
          "Set num_channels=20 in the finetune data_module.")


if __name__ == "__main__":
    main()

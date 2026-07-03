#*----------------------------------------------------------------------------*
#* TUH EEG Epilepsy (TUEP, v3.0.0) preprocessing for the LuMamba pretrained model.
#*
#* Mirrors process_raw_eeg.py (tuab / file-level binary label) but adapted to the TUH
#* EEG Epilepsy corpus and to feed the LuMamba / LUNA finetune dataset (datasets.tuh_dataset.
#* TUH_Dataset). Per recording: read EDF -> rename LE->REF -> band-pass 0.1-75 Hz -> notch
#* 60 Hz -> resample 256 Hz -> TCP bipolar (the 20 NON-ear pairs, in CHN_ORDER order) ->
#* 5 s windows -> one {"X": (20, 1280), "y": label} pickle per window. Label is per-patient
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
#* Then point config/experiment/LuMamba_finetune.yaml at:
#*   data_module.{train,val,test}._target_: datasets.tuh_dataset.TUH_Dataset
#*   data_module.{train,val,test}.hdf5_file:  <output_dir>/TUEP_data/{train,val,test}.h5
#*   data_module.{train,val,test}.num_channels: 20   (CHN_ORDER[:20])
#*----------------------------------------------------------------------------*

import argparse
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hdf5 import create_hdf5  # noqa: E402  (sibling script, bundles pkls -> .h5)

# --- The 20 non-ear TCP bipolar pairs, in TUH_Dataset.CHN_ORDER order (ears dropped). ---
# Each value is the (anode, cathode) referential channel pair, in REF names (LE recordings
# are renamed LE->REF first, so the same pairs apply to all four TUH montages).
BIPOLAR_PAIRS = [
    ("FP1-F7", "EEG FP1-REF", "EEG F7-REF"), ("F7-T3", "EEG F7-REF", "EEG T3-REF"),
    ("T3-T5", "EEG T3-REF", "EEG T5-REF"),   ("T5-O1", "EEG T5-REF", "EEG O1-REF"),
    ("FP2-F8", "EEG FP2-REF", "EEG F8-REF"), ("F8-T4", "EEG F8-REF", "EEG T4-REF"),
    ("T4-T6", "EEG T4-REF", "EEG T6-REF"),   ("T6-O2", "EEG T6-REF", "EEG O2-REF"),
    ("T3-C3", "EEG T3-REF", "EEG C3-REF"),   ("C3-CZ", "EEG C3-REF", "EEG CZ-REF"),
    ("CZ-C4", "EEG CZ-REF", "EEG C4-REF"),   ("C4-T4", "EEG C4-REF", "EEG T4-REF"),
    ("FP1-F3", "EEG FP1-REF", "EEG F3-REF"), ("F3-C3", "EEG F3-REF", "EEG C3-REF"),
    ("C3-P3", "EEG C3-REF", "EEG P3-REF"),   ("P3-O1", "EEG P3-REF", "EEG O1-REF"),
    ("FP2-F4", "EEG FP2-REF", "EEG F4-REF"), ("F4-C4", "EEG F4-REF", "EEG C4-REF"),
    ("C4-P4", "EEG C4-REF", "EEG P4-REF"),   ("P4-O2", "EEG P4-REF", "EEG O2-REF"),
]
N_BIPOLAR = len(BIPOLAR_PAIRS)  # 20
SFREQ = 256
WINDOW_SAMPLES = 5 * SFREQ      # 5 s at 256 Hz -> 1280
COHORT_LABEL = {"00_epilepsy": 1, "01_no_epilepsy": 0}


def make_bipolar_20(raw):
    """(20, T) float array in CHN_ORDER[:20] order, or None if any of the 20 pairs is
    underivable (a required referential channel is absent)."""
    names = raw.ch_names
    data = raw.get_data(units="uV")
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for _name, ch1, ch2 in BIPOLAR_PAIRS:
        if ch1 not in idx or ch2 not in idx:
            return None
        out.append(data[idx[ch1]] - data[idx[ch2]])
    return np.asarray(out, dtype=np.float32)


def process_and_dump_file(params):
    """Worker: preprocess one EDF and dump its 5 s windows as pickles."""
    file_path, dump_folder, label, max_windows, min_duration_s = params
    stem = os.path.basename(file_path).split(".")[0]
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        if any("-LE" in ch for ch in raw.ch_names):
            raw.rename_channels(lambda x: x.replace("-LE", "-REF"))
        # keep only the referential channels the bipolar montage needs (relaxed: '_a'
        # montages are missing the ear channels, which we do not use anyway)
        needed = {c for _n, a, b in BIPOLAR_PAIRS for c in (a, b)}
        present = [c for c in raw.ch_names if c in needed]
        if len(present) < len(needed):
            raise ValueError(f"missing referential channels ({len(present)}/{len(needed)})")
        raw.pick(present)

        # Drop recordings shorter than min_duration_s (native rate, before filtering): the
        # 0.1 Hz high-pass needs a ~68 s FIR filter, so shorter signals get edge distortion
        # (RuntimeWarning "filter_length ... longer than the signal").
        if min_duration_s and raw.n_times / float(raw.info["sfreq"]) < min_duration_s:
            return

        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(60, verbose=False)
        if int(round(raw.info["sfreq"])) != SFREQ:
            raw.resample(SFREQ, npad="auto", n_jobs=1, verbose=False)

        data = make_bipolar_20(raw)
        if data is None or data.shape[0] != N_BIPOLAR:
            raise ValueError("could not build the 20 bipolar channels")
        n_times = data.shape[1]
        n_win = n_times // WINDOW_SAMPLES
        if max_windows is not None:
            n_win = min(n_win, max_windows)
        for i in range(n_win):
            seg = data[:, i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]
            with open(os.path.join(dump_folder, f"{stem}_{i}.pkl"), "wb") as f:
                pickle.dump({"X": seg, "y": int(label)}, f)
    except Exception as e:  # noqa: BLE001
        with open("tuep-process-errors.txt", "a") as f:
            f.write(f"Error processing {file_path}: {e}\n")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", required=True, help="Path to the TUH EEG Epilepsy v3.0.0 dir (has 00_epilepsy / 01_no_epilepsy).")
    parser.add_argument("--output_dir", required=True, help="Where TUEP_data/{processed,*.h5} is written.")
    parser.add_argument("--processes", type=int, default=24, help="Parallel worker processes (default 24).")
    parser.add_argument("--interictal", action="store_true", help="Drop recordings with a seizure annotation (.csv_bi 'seiz'); diagnosis task.")
    parser.add_argument("--max_windows_per_recording", type=int, default=None, help="Cap 5 s windows per recording (default: all).")
    parser.add_argument("--min_duration_s", type=float, default=0.0,
                        help="Drop recordings shorter than this many seconds (0 = keep all). Use ~70+ to avoid the "
                        "0.1 Hz filter distorting short recordings; e.g. 120 for >= 2 min.")
    parser.add_argument("--seed", type=int, default=42, help="Subject-split seed (default 42).")
    parser.add_argument("--keep_pkl", action="store_true", help="Keep the intermediate .pkl files after building the .h5 files.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    base = os.path.join(args.output_dir, "TUEP_data")
    proc = os.path.join(base, "processed")
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(proc, split), exist_ok=True)

    recs = collect_recordings(root_dir, args.interictal)
    if not recs:
        raise SystemExit(f"No EDF recordings found under {root_dir}.")
    splits = split_subjects(recs, args.seed)
    subj_to_split = {s: sp for sp, subs in splits.items() for s in subs}
    print(f"Recordings: {len(recs)} | subjects: {len(subj_to_split)} "
          f"(train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])})")

    params = [
        (str(edf), os.path.join(proc, subj_to_split[subject]), label,
         args.max_windows_per_recording, args.min_duration_s)
        for edf, subject, label in recs
    ]
    print(f"Processing {len(params)} recordings with {args.processes} processes...")
    with Pool(processes=args.processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_and_dump_file, params), total=len(params)))

    for split in ("train", "val", "test"):
        src = os.path.join(proc, split)
        tgt = os.path.join(base, f"{split}.h5")
        create_hdf5(src, tgt, finetune=True)
    if not args.keep_pkl:
        import shutil
        shutil.rmtree(proc, ignore_errors=True)
    print(f"Done. HDF5 at {base}/{{train,val,test}}.h5 "
          f"(X: (N, {N_BIPOLAR}, {WINDOW_SAMPLES}), y: 0/1). Set num_channels=20 in the finetune data_module.")


if __name__ == "__main__":
    main()

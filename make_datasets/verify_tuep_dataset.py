#*----------------------------------------------------------------------------*
#* Audit the TUEP dataset: window labels + train/val/test split integrity.
#*
#* Reproduces the recording -> subject -> label -> split assignment that
#* process_tuep_eeg.py uses (the same collect_recordings + split_subjects, same seed),
#* and checks the invariants a chance-level classifier would violate if broken:
#*   (1) every subject carries exactly ONE cohort label (epilepsy XOR no-epilepsy),
#*   (2) the splits are subject-disjoint (no subject leaks across train/val/test),
#*   (3) each split is stratified (both classes present, similar epilepsy fraction).
#* Then, if --output_dir is given, it cross-checks the actual {train,val,test}.h5 window
#* labels against the source assignment (both classes present per split, sensible ratios).
#*
#* Pass the SAME label/split-affecting flags used to build the data (--interictal, --seed);
#* --min_duration_s / --max_windows_per_subject only change window COUNTS, not label/split,
#* so they are not needed here. Run on the HPC (needs the corpus), from make_datasets/:
#*
#*   python verify_tuep_dataset.py --root_dir /path/to/tuh_eeg_epilepsy/v3.0.0 \
#*       --interictal --output_dir /work/.../BioFoundation
#*----------------------------------------------------------------------------*

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from process_tuep_eeg import collect_recordings, split_subjects

LABEL_NAME = {0: "no_epilepsy", 1: "epilepsy"}
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", required=True, help="TUH EEG Epilepsy v3.0.0 dir (00_epilepsy / 01_no_epilepsy).")
    parser.add_argument("--interictal", action="store_true", help="Match the build: drop seizure recordings.")
    parser.add_argument("--seed", type=int, default=42, help="Match the build's subject-split seed (default 42).")
    parser.add_argument("--output_dir", default=None,
                        help="If set, cross-check <output_dir>/TUEP_data/{train,val,test}.h5.")
    args = parser.parse_args()

    recs = collect_recordings(Path(args.root_dir), args.interictal)
    if not recs:
        raise SystemExit(f"No EDF recordings found under {args.root_dir}.")

    warnings = []

    # (1) subject -> label consistency: a subject must sit under exactly one cohort.
    subj_labels = defaultdict(set)
    for _edf, subject, label in recs:
        subj_labels[subject].add(label)
    mixed = {s: v for s, v in subj_labels.items() if len(v) > 1}
    subj_label = {s: next(iter(v)) for s, v in subj_labels.items()}
    cls_counts = Counter(subj_label.values())
    print(f"Recordings: {len(recs)} | subjects: {len(subj_label)} "
          f"(no_epilepsy={cls_counts[0]}, epilepsy={cls_counts[1]})")
    if mixed:
        warnings.append(f"{len(mixed)} subjects have MIXED cohort labels: {list(mixed)[:5]}")
        print(f"  [FAIL] {len(mixed)} subjects appear under both cohorts (should be 0): {list(mixed)[:5]}")
    else:
        print("  [OK] every subject has exactly one cohort label.")

    # (2) split assignment + disjointness (no subject in more than one split -> no leakage).
    splits = split_subjects(recs, args.seed)
    subj_to_split = {s: sp for sp, subs in splits.items() for s in subs}
    leak = [s for s in subj_label if sum(s in splits[sp] for sp in SPLITS) != 1]
    unassigned = [s for s in subj_label if s not in subj_to_split]
    if leak:
        warnings.append(f"{len(leak)} subjects in >1 split (LEAKAGE): {leak[:5]}")
        print(f"  [FAIL] {len(leak)} subjects assigned to more than one split (LEAKAGE): {leak[:5]}")
    elif unassigned:
        warnings.append(f"{len(unassigned)} subjects unassigned to any split: {unassigned[:5]}")
        print(f"  [FAIL] {len(unassigned)} subjects not assigned to any split: {unassigned[:5]}")
    else:
        print("  [OK] splits are subject-disjoint and cover every subject (no leakage).")

    # (3) stratification: both classes present per split, epilepsy fraction close to overall.
    rec_split_class = Counter((subj_to_split[s], label) for _e, s, label in recs)
    overall_frac = cls_counts[1] / max(1, sum(cls_counts.values()))
    print("\nPer-split class balance (subject-level split; window counts scale with these):")
    for sp in SPLITS:
        subs = splits[sp]
        n0 = sum(1 for s in subs if subj_label[s] == 0)
        n1 = sum(1 for s in subs if subj_label[s] == 1)
        r0, r1 = rec_split_class[(sp, 0)], rec_split_class[(sp, 1)]
        frac = n1 / max(1, n0 + n1)
        flag = "" if (n0 > 0 and n1 > 0) else "  [FAIL] a class is missing from this split"
        if not (n0 > 0 and n1 > 0):
            warnings.append(f"split '{sp}' is missing a class")
        print(f"  {sp:5s}: subjects no_epilepsy={n0:3d} epilepsy={n1:3d} (epilepsy frac {frac:.2f}) | "
              f"recordings no_epilepsy={r0:4d} epilepsy={r1:4d}{flag}")
    print(f"  overall epilepsy fraction: {overall_frac:.2f} (each split's fraction should be close to this)")

    # HDF5 cross-check: the actual window labels per split (both classes, sensible ratio).
    if args.output_dir:
        import h5py
        import numpy as np
        base = Path(args.output_dir) / "TUEP_data"
        print("\nHDF5 window-label distribution (actual bundled data):")
        for sp in SPLITS:
            f = base / f"{sp}.h5"
            if not f.exists():
                print(f"  {sp:5s}: {f} missing")
                warnings.append(f"{f} missing")
                continue
            with h5py.File(f, "r") as d:
                ys = np.concatenate([d[k]["y"][:] for k in d]) if len(d) else np.array([])
            u, c = np.unique(ys, return_counts=True)
            dist = {LABEL_NAME[int(k)]: int(v) for k, v in zip(u, c)}
            n = int(c.sum()) if len(c) else 0
            frac = (dist.get("epilepsy", 0) / n) if n else 0.0
            ok = len(u) == 2
            if not ok:
                warnings.append(f"HDF5 split '{sp}' has only one class")
            print(f"  {sp:5s}: {dist}  (n={n}, epilepsy frac {frac:.2f})  "
                  f"{'[OK] both classes' if ok else '[FAIL] only one class present'}")

    print("\n" + ("ALL CHECKS PASSED." if not warnings else f"{len(warnings)} ISSUE(S) FOUND:"))
    for w in warnings:
        print(f"  - {w}")
    raise SystemExit(1 if warnings else 0)


if __name__ == "__main__":
    main()

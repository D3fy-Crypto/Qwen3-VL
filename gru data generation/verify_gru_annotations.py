#!/usr/bin/env python3
"""Verify GRU action-prefix generation for navigation annotations.

By default this recomputes ``gru`` from the *plain* annotations (via
nav_action_encoding.build_trajectory_gru) and runs three layers of checks:

  1. Structural (all datasets): every row has a ``gru``; it ends in ``0``; every
     entry is in {0,1,2,3}; ``0`` occurs only as the final element (no interior 0).
  2. Oracle cross-check (R2R, RxR): each step's own codes parsed from ``a`` equal the
     colleague's ``action_dict_all.json`` body (``action_dict_all[vid][:-1]``).
  3. Anchor (R2R only): recomputed ``gru`` matches the existing, verified on-disk
     ``R2R/annotations_with_gru.json`` exactly (0 diffs).

If ``--generated FILE`` is given, the file's stored ``gru`` is additionally asserted
to equal the recomputed values (proves the generator wrote what the rule says).

Exit code is non-zero if any hard check fails.

Examples:
    python verify_gru_annotations.py --dataset r2r
    python verify_gru_annotations.py --dataset rxr --generated generated/rxr_annotations_with_gru.json
    python verify_gru_annotations.py --dataset human
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav_action_encoding import (  # noqa: E402
    accumulate_episodes,
    action_codes_from_answer,
    build_trajectory_gru,
    episodes_to_step_gru,
    split_video_id,
)

NAVILA_CANDIDATES = [
    "/opt/IROS_proj/NaVILA-Dataset",
    "/home/rithvik/IROS_proj/NaVILA-Dataset",
    "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset",
]

DATASET_INPUT = {
    "r2r": "R2R/annotations.json",
    "rxr": "RxR/annotations.json",
    "human": "Human/annotations.json",
}
DATASET_OVERSAMPLED = {
    "r2r": "R2R/annotations_oversampled.json",
    "rxr": "RxR/annotations_oversampled.json",
    "human": "Human/annotations_oversampled.json",
}
# action_dict_all locations (R2R's lives at the dataset root; Human has none).
ACTION_DICT = {
    "r2r": "action_dict_all.json",
    "rxr": "RxR/action_dict_all.json",
}
ANCHOR = {
    "r2r": "R2R/annotations_with_gru.json",
}
VALID_CODES = {0, 1, 2, 3}


def detect_navila_base(override=None):
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"--navila-base not found: {override}")
        return override
    for cand in NAVILA_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("No NaVILA-Dataset base found.")


def episode_stats(records):
    traj_steps = defaultdict(list)
    vid_counts = defaultdict(int)
    for r in records:
        vid = str(r.get("video_id", ""))
        vid_counts[vid] += 1
        t, s = split_video_id(vid)
        traj_steps[t].append(s)
    dup_vids = sum(1 for v, c in vid_counts.items() if c > 1)
    gaps = 0
    for t, ss in traj_steps.items():
        uniq = sorted(set(ss))
        if uniq != list(range(uniq[0], uniq[0] + len(uniq))) or uniq[0] != 0:
            gaps += 1
    return len(traj_steps), dup_vids, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_INPUT), required=True)
    ap.add_argument("--navila-base", default=None)
    ap.add_argument("--input", default=None, help="plain annotations.json (default from --dataset)")
    ap.add_argument("--generated", default=None, help="generated *_with_gru.json to validate")
    ap.add_argument("--oversampled", action="store_true",
                    help="union the oversampled file into the recompute base so the "
                         "checks cover every oversampled row (matches generator's map)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = detect_navila_base(args.navila_base)
    in_path = args.input or os.path.join(base, DATASET_INPUT[args.dataset])
    print(f"[verify] dataset : {args.dataset}")
    print(f"[verify] input   : {in_path}")

    t0 = time.time()
    with open(in_path) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    print(f"[verify] loaded {len(records):,} records in {time.time()-t0:.1f}s")

    n_traj, dup_vids, gaps = episode_stats(records)
    print(f"[verify] episodes={n_traj:,} | duplicate video_ids={dup_vids:,} | "
          f"episodes with gaps/non-zero-start={gaps:,}")

    # Recompute gru. Base = plain annotations (source of truth for the rule). In
    # --oversampled mode, union the oversampled file too so recomputed covers every
    # oversampled video_id (identical map to the generator).
    episodes = accumulate_episodes(records)
    if args.oversampled and not args.limit:
        over_path = os.path.join(base, DATASET_OVERSAMPLED[args.dataset])
        if Path(over_path).exists():
            with open(over_path) as f:
                over_records = json.load(f)
            accumulate_episodes(over_records, episodes)
            del over_records
            print(f"[verify] unioned oversampled {over_path}")
    step_gru = episodes_to_step_gru(episodes)
    recomputed = {}
    for rec in records:
        vid = str(rec.get("video_id", ""))
        traj, step = split_video_id(vid)
        recomputed[vid] = step_gru[traj][step]
    # keep step_gru for --generated coverage of oversampled-only vids
    _step_gru_full = step_gru

    failures = []

    # ---- 1. Structural checks ----
    interior_zero = 0
    bad_tail = 0
    bad_code = 0
    max_len = 0
    for vid, g in recomputed.items():
        if not g or g[-1] != 0:
            bad_tail += 1
        if 0 in g[:-1]:
            interior_zero += 1
        if any(c not in VALID_CODES for c in g):
            bad_code += 1
        max_len = max(max_len, len(g))
    print(f"[verify] structural: max_gru_len={max_len} | not_ending_in_0={bad_tail} | "
          f"interior_zero={interior_zero} | out_of_range_codes={bad_code}")
    if bad_tail:
        failures.append(f"{bad_tail} gru not ending in 0")
    if interior_zero:
        failures.append(f"{interior_zero} gru with an interior 0")
    if bad_code:
        failures.append(f"{bad_code} gru with out-of-range codes")

    # ---- 2. Oracle cross-check vs action_dict_all.json ----
    if args.dataset in ACTION_DICT:
        ad_path = os.path.join(base, ACTION_DICT[args.dataset])
        if Path(ad_path).exists():
            ad = json.load(open(ad_path))
            mism = 0
            shown = 0
            for r in records:
                vid = str(r.get("video_id", ""))
                if vid not in ad:
                    continue
                expected_body = list(ad[vid][:-1]) if ad[vid] and ad[vid][-1] == 0 else list(ad[vid])
                got = action_codes_from_answer(str(r.get("a", "")))
                if got != expected_body:
                    mism += 1
                    if shown < 3:
                        print(f"   [oracle MISMATCH] {vid}: answer_codes={got} vs "
                              f"action_dict_all={expected_body}")
                        shown += 1
            print(f"[verify] action_dict_all cross-check: mismatches={mism} "
                  f"(of {len(ad):,} oracle entries)")
            if mism:
                failures.append(f"{mism} action_dict_all mismatches")
        else:
            print(f"[verify] action_dict_all not found ({ad_path}); skipping oracle check")

    # ---- 3. Anchor: must match the existing on-disk with_gru file exactly ----
    if args.dataset in ANCHOR and not args.limit:
        anc_path = os.path.join(base, ANCHOR[args.dataset])
        if Path(anc_path).exists():
            t0 = time.time()
            anc = json.load(open(anc_path))
            diffs = 0
            shown = 0
            covered = 0
            for r in anc:
                vid = str(r.get("video_id", ""))
                if vid not in recomputed:
                    continue
                covered += 1
                if [int(x) for x in r.get("gru", [])] != recomputed[vid]:
                    diffs += 1
                    if shown < 3:
                        print(f"   [anchor DIFF] {vid}: ondisk={r.get('gru')[:8]} "
                              f"recomputed={recomputed[vid][:8]}")
                        shown += 1
            print(f"[verify] anchor vs on-disk {ANCHOR[args.dataset]}: covered={covered:,} "
                  f"diffs={diffs} ({time.time()-t0:.1f}s)")
            if diffs:
                failures.append(f"{diffs} anchor diffs vs on-disk file")
        else:
            print(f"[verify] anchor file not found ({anc_path}); skipping")

    # ---- optional: validate a generated file matches the recompute ----
    if args.generated:
        gen = json.load(open(args.generated))
        if args.limit:
            gen = gen[: args.limit]
        diffs = 0
        shown = 0
        uncovered = 0
        struct_bad = 0
        for r in gen:
            vid = str(r.get("video_id", ""))
            g = [int(x) for x in r.get("gru", [])]
            # structural checks on the FILE's own gru. NOTE: len(frames) is
            # intentionally NOT checked — gru length (number of history actions) is
            # unrelated to how many frames a slice stores.
            if not g or g[-1] != 0 or 0 in g[:-1] or any(c not in VALID_CODES for c in g):
                struct_bad += 1
            # expected gru: prefer plain recompute; fall back to full (union) map so
            # oversampled-only trajectories are still covered.
            exp = recomputed.get(vid)
            if exp is None:
                traj, step = split_video_id(vid)
                exp = _step_gru_full.get(traj, {}).get(step)
            if exp is None:
                uncovered += 1
                continue
            if g != exp:
                diffs += 1
                if shown < 3:
                    print(f"   [generated DIFF] {vid}: file={g[:10]} recomputed={exp[:10]}")
                    shown += 1
        print(f"[verify] generated file {Path(args.generated).name}: rows={len(gen):,} "
              f"diffs={diffs} uncovered={uncovered} struct_bad={struct_bad}")
        if diffs:
            failures.append(f"{diffs} generated-file diffs")
        if uncovered:
            failures.append(f"{uncovered} generated rows with no reconstructable gru")
        if struct_bad:
            failures.append(f"{struct_bad} generated rows failing structural check")

    print("=" * 60)
    if failures:
        print("[verify] FAILED:")
        for fdesc in failures:
            print("   - " + fdesc)
        sys.exit(1)
    print("[verify] ALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()

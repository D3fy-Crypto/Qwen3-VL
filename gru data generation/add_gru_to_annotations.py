#!/usr/bin/env python3
"""Add a per-record ``gru`` action-prefix field to navigation annotations.

Reconstructs each episode from the exploded per-step rows and writes a copy of the
annotations with a ``gru`` list added to every record (see nav_action_encoding.py
for the exact rule). R2R / RxR / Human share the ``{video_id, q, a, frames}`` schema.

EnvDrop is intentionally NOT handled here: its ``envdrop_motion.json`` already carries
an inline per-frame ``motion`` list (a different, description-task scheme).

Outputs go to ``--out-dir`` (default ``./generated`` next to this script), which is
git-ignored — these files are for local verification, not for committing.

Examples:
    python "add_gru_to_annotations.py" --dataset r2r          # regenerate the anchor
    python "add_gru_to_annotations.py" --dataset rxr
    python "add_gru_to_annotations.py" --dataset human --limit 5000   # quick sample
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from anywhere: the shared module sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav_action_encoding import (  # noqa: E402
    accumulate_episodes,
    episodes_to_step_gru,
    split_video_id,
)

# Same auto-detect order as qwen-vl-finetune/qwenvl/data/__init__.py.
NAVILA_CANDIDATES = [
    "/opt/IROS_proj/NaVILA-Dataset",
    "/home/rithvik/IROS_proj/NaVILA-Dataset",
    "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset",
]

# dataset -> relative path of the plain annotations under NAVILA_BASE.
DATASET_INPUT = {
    "r2r": "R2R/annotations.json",
    "rxr": "RxR/annotations.json",
    "human": "Human/annotations.json",
}
# dataset -> relative path of the oversampled annotations (what training loads).
DATASET_OVERSAMPLED = {
    "r2r": "R2R/annotations_oversampled.json",
    "rxr": "RxR/annotations_oversampled.json",
    "human": "Human/annotations_oversampled.json",
}


def detect_navila_base(override: str = None) -> str:
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"--navila-base not found: {override}")
        return override
    for cand in NAVILA_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "No NaVILA-Dataset base found. Tried: " + ", ".join(NAVILA_CANDIDATES)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_INPUT), required=True)
    ap.add_argument("--navila-base", default=None, help="override NaVILA-Dataset root")
    ap.add_argument("--input", default=None, help="explicit annotations.json (overrides --dataset path)")
    ap.add_argument("--oversampled", action="store_true",
                    help="use annotations_oversampled.json as the records to emit; "
                         "the trajectory map is still built from the UNION of the plain "
                         "+ oversampled files so exclusive prefixes stay complete")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "generated"))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N rows (quick check)")
    ap.add_argument("--indent", type=int, default=0, help="JSON indent (0 = compact, matches on-disk format)")
    args = ap.parse_args()

    base = detect_navila_base(args.navila_base)

    # `in_path` = the records we emit (with gru attached). In oversampled mode this is
    # the oversampled file; the plain file is loaded only to complete the trajectory map.
    if args.input:
        in_path = args.input
    elif args.oversampled:
        in_path = os.path.join(base, DATASET_OVERSAMPLED[args.dataset])
    else:
        in_path = os.path.join(base, DATASET_INPUT[args.dataset])
    if not Path(in_path).exists():
        raise FileNotFoundError(f"Input annotations not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit else ""
    stem = f"{args.dataset}_oversampled_with_gru" if args.oversampled else f"{args.dataset}_annotations_with_gru"
    out_path = out_dir / f"{stem}{suffix}.json"

    print(f"[gen] dataset   : {args.dataset}")
    print(f"[gen] navila    : {base}")
    print(f"[gen] oversampled: {args.oversampled}")
    print(f"[gen] input     : {in_path}")
    print(f"[gen] output    : {out_path}")

    t0 = time.time()
    with open(in_path) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    print(f"[gen] loaded {len(records):,} records in {time.time()-t0:.1f}s")

    # Build the trajectory->step->codes map. In oversampled mode we UNION the plain
    # annotations first so every step of a trajectory is present even if oversampling
    # dropped some — the per-video_id gru is deterministic, so duplicates are harmless.
    episodes = accumulate_episodes(records)
    if args.oversampled and not args.input:
        plain_path = os.path.join(base, DATASET_INPUT[args.dataset])
        if Path(plain_path).exists():
            t1 = time.time()
            with open(plain_path) as f:
                plain_records = json.load(f)
            before = sum(len(v) for v in episodes.values())
            accumulate_episodes(plain_records, episodes)
            after = sum(len(v) for v in episodes.values())
            del plain_records
            print(f"[gen] unioned plain {plain_path} in {time.time()-t1:.1f}s "
                  f"(+{after-before:,} steps not in oversampled)")
        else:
            print(f"[gen] WARNING: plain file not found ({plain_path}); "
                  f"reconstructing from oversampled alone (gaps possible)")

    step_gru = episodes_to_step_gru(episodes)

    # Attach gru to each emitted row in place (preserving key + record order).
    # gru length (= number of history actions in the trajectory so far) has NO
    # relationship to len(frames); the two are independent and are not compared.
    n_traj = len(episodes)
    max_len = 0
    for rec in records:
        traj, step = split_video_id(str(rec.get("video_id", "")))
        g = step_gru[traj][step]
        rec["gru"] = g
        if len(g) > max_len:
            max_len = len(g)
    print(f"[gen] reconstructed {n_traj:,} episodes | max gru len = {max_len}")

    dump_kwargs = dict(ensure_ascii=False)
    if args.indent and args.indent > 0:
        dump_kwargs["indent"] = args.indent
    else:
        dump_kwargs["separators"] = (",", ":")  # compact, matches on-disk file
    t0 = time.time()
    with open(out_path, "w") as f:
        json.dump(records, f, **dump_kwargs)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[gen] wrote {out_path.name} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
    print("[gen] done. Verify with: python verify_gru_annotations.py --dataset "
          f"{args.dataset} --generated {out_path}")


if __name__ == "__main__":
    main()

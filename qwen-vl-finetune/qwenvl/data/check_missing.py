"""
Check what fraction of data points have missing image/video files per dataset.
Usage: python qwen-vl-finetune/qwenvl/data/check_missing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwenvl.data import data_dict as DATASETS


def check_dataset(name, cfg):
    ann_path = Path(cfg["annotation_path"])
    data_path = Path(cfg["data_path"])

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")

    if not ann_path.exists():
        print(f"  [SKIP] annotation not found: {ann_path}")
        return

    print(f"  Scanning files in {data_path} ...")
    existing = set(str(p.relative_to(data_path)) for p in data_path.rglob("*") if p.is_file())
    print(f"  Found {len(existing)} files on disk.")

    with open(ann_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = list(data.values())

    total = len(data)
    missing_items = 0
    missing_files = 0
    first_missing = []

    for item in data:
        if "frames" not in item:
            # ScanQA: check video file
            rel = f"{item['video_id']}.mp4"
            if rel not in existing:
                missing_items += 1
                missing_files += 1
                if len(first_missing) < 3:
                    first_missing.append(str(data_path / rel))
        else:
            # Navigation: check all frame files
            bad = [f for f in item["frames"] if f not in existing]
            if bad:
                missing_items += 1
                missing_files += len(bad)
                if len(first_missing) < 3:
                    first_missing.append(str(data_path / bad[0]))

    pct = 100.0 * missing_items / total if total else 0
    print(f"  Total items  : {total}")
    print(f"  Items w/ missing files: {missing_items} ({pct:.1f}%)")
    print(f"  Total missing files   : {missing_files}")
    if first_missing:
        print(f"  Examples of missing:")
        for p in first_missing:
            print(f"    {p}")


def main():
    for name, cfg in DATASETS.items():
        check_dataset(name, cfg)
    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()

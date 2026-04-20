#!/usr/bin/env python3
"""
Convert R2R annotations into GRU-ready format by adding a `gru` action sequence field.

Input schema (list[dict]):
  {
    "video_id": "914-23",
    "q": "...",
    "a": "The next action is move forward 75 cm.",
    "frames": [...]
  }

Output schema adds:
  "gru": [int, int, ...]

Action IDs:
  0 = STOP
  1 = FORWARD
  2 = TURN_LEFT
  3 = TURN_RIGHT
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

STOP = 0
FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

DEGREE_PATTERN = re.compile(r"(\d+)\s*degree", re.IGNORECASE)
CM_PATTERN = re.compile(r"(\d+)\s*cm", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add GRU action sequences to R2R annotations.")
    parser.add_argument(
        "--input",
        default="/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/annotations.json",
        help="Path to source R2R annotations JSON.",
    )
    parser.add_argument(
        "--output",
        default="/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/annotations_with_gru.json",
        help="Path to output JSON with `gru` field.",
    )
    parser.add_argument(
        "--mode",
        choices=["cumulative", "step"],
        default="cumulative",
        help=(
            "cumulative: gru is history up to this step (recommended). "
            "step: gru contains only this sample action(s)."
        ),
    )
    parser.add_argument(
        "--append-stop",
        action="store_true",
        default=True,
        help="Append STOP (0) token at the end of each gru sequence.",
    )
    parser.add_argument(
        "--no-append-stop",
        action="store_false",
        dest="append_stop",
        help="Do not append STOP token.",
    )
    return parser.parse_args()


def split_video_id(video_id: str) -> Tuple[str, int]:
    """Split '<traj>-<step>' into ('traj', step)."""
    if not isinstance(video_id, str) or "-" not in video_id:
        return str(video_id), 0
    traj, step = video_id.rsplit("-", 1)
    try:
        return traj, int(step)
    except ValueError:
        return traj, 0


def action_codes_from_answer(answer: str) -> List[int]:
    """Map natural-language answer to integer action code sequence."""
    answer = (answer or "").lower()

    if "right" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        steps = max(1, steps)
        return [TURN_RIGHT] * steps

    if "left" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        steps = max(1, steps)
        return [TURN_LEFT] * steps

    if "move forward" in answer or "forward" in answer:
        match = CM_PATTERN.search(answer)
        steps = int(match.group(1)) // 25 if match else 1
        steps = max(1, steps)
        return [FORWARD] * steps

    # Unknown patterns are treated as no-op; STOP can be appended later.
    return []


def build_gru_sequences(annotations: List[dict], mode: str, append_stop: bool) -> Dict[str, List[int]]:
    by_traj: Dict[str, List[Tuple[int, str, List[int]]]] = defaultdict(list)
    for ann in annotations:
        vid = ann.get("video_id", "")
        traj, step = split_video_id(vid)
        codes = action_codes_from_answer(ann.get("a", ""))
        by_traj[traj].append((step, vid, codes))

    gru_by_video_id: Dict[str, List[int]] = {}
    for traj, triples in by_traj.items():
        triples.sort(key=lambda x: x[0])
        running: List[int] = []

        for _, vid, codes in triples:
            if mode == "cumulative":
                running.extend(codes)
                seq = list(running)
            else:
                seq = list(codes)

            if append_stop:
                seq = seq + [STOP]

            # Ensure non-empty sequence for model compatibility.
            if not seq:
                seq = [STOP]

            gru_by_video_id[vid] = seq

    return gru_by_video_id


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    annotations = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError("Expected top-level JSON list of annotations.")

    gru_by_video_id = build_gru_sequences(
        annotations=annotations,
        mode=args.mode,
        append_stop=args.append_stop,
    )

    output: List[dict] = []
    lengths: List[int] = []
    missing = 0
    for ann in annotations:
        vid = ann.get("video_id", "")
        gru = gru_by_video_id.get(vid)
        if gru is None:
            missing += 1
            gru = [STOP]
        row = dict(ann)
        row["gru"] = gru
        output.append(row)
        lengths.append(len(gru))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if lengths:
        print(f"Wrote: {output_path}")
        print(f"Samples: {len(output)}")
        print(f"Mode: {args.mode}")
        print(f"append_stop: {args.append_stop}")
        print(f"GRU length min/mean/max: {min(lengths)}/{sum(lengths)/len(lengths):.2f}/{max(lengths)}")
        print(f"Missing video_id mappings: {missing}")
        print("Example:")
        ex = output[0]
        print({"video_id": ex.get("video_id"), "a": ex.get("a"), "gru_head": ex.get("gru", [])[:16]})


if __name__ == "__main__":
    main()

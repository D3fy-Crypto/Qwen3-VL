"""Shared, dependency-free helpers for building GRU action prefixes.

This module reconstructs each navigation *episode* from the exploded per-step
annotation rows (R2R / RxR / Human all use ``video_id = "<traj>-<step>"``) and
emits the per-row ``gru`` action-code list the GRU-Qwen pipeline consumes.

The two parsing primitives below are copied verbatim from the training pipeline
(`qwen-vl-finetune/qwenvl/data/data_processor_gru_chang.py`, `action_codes_from_answer`
and `split_video_id`) so generation and training agree on a single encoding.
`verify_gru_annotations.py` pins this equivalence by reproducing the existing,
verified on-disk `R2R/annotations_with_gru.json` exactly (0 diffs / 288,594 rows).

Action codes:  STOP=0  FORWARD=1  TURN_LEFT=2  TURN_RIGHT=3

`gru(step S) = concat(action_codes(a) for every prior step k < S) + [0]`

The trailing ``0`` is a *structural placeholder for the current frame/node* (the
action still to be predicted), NOT a STOP the agent emitted: it is present on every
row regardless of the answer, which is what makes ``len(gru) == len(frames)``. A real
STOP answer parses to ``[]`` and contributes nothing, so genuine stops never add an
interior ``0`` — the only ``0`` is the structural tail.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Mirror data_processor_gru_chang.py.
STOP = 0
FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

DEGREE_PATTERN = re.compile(r"(\d+)\s*degree", re.IGNORECASE)
CM_PATTERN = re.compile(r"(\d+)\s*cm", re.IGNORECASE)


def action_codes_from_answer(answer: str) -> List[int]:
    """Parse a navigation answer into repeated action codes.

    Verbatim copy of `data_processor_gru_chang.py:action_codes_from_answer`.
    e.g. "turn right 45 degree" -> [3, 3, 3]; "move forward 75 cm" -> [1, 1, 1];
    "...stop..." -> [].
    """
    answer = (answer or "").lower()

    if "right" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        return [TURN_RIGHT] * max(1, steps)

    if "left" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        return [TURN_LEFT] * max(1, steps)

    if "move forward" in answer or "forward" in answer:
        match = CM_PATTERN.search(answer)
        steps = int(match.group(1)) // 25 if match else 1
        return [FORWARD] * max(1, steps)

    return []


def split_video_id(video_id: str) -> Tuple[str, int]:
    """Split ``"<traj>-<step>"`` into ``(traj, step)``.

    Verbatim copy of `data_processor_gru_chang.py:split_video_id`. Only the trailing
    ``-<step>`` is split, so hyphens inside the trajectory id are preserved
    (e.g. Human ``"-23esP--xK8_0-0"`` -> ``("-23esP--xK8_0", 0)``).
    """
    if not isinstance(video_id, str) or "-" not in video_id:
        return str(video_id), 0
    traj, step = video_id.rsplit("-", 1)
    try:
        return traj, int(step)
    except ValueError:
        return traj, 0


def build_trajectory_gru(records: List[dict]) -> Dict[str, List[int]]:
    """Reconstruct episodes and return ``{video_id: gru}`` for every row.

    Steps:
      1. group rows by trajectory via ``split_video_id``;
      2. per *unique* step, parse its own action codes from ``a`` (duplicate rows
         for the same step — e.g. Human's 5x stop-step — collapse to one entry);
      3. walk steps in ascending order with a running prefix, snapshotting
         ``gru = running + [0]`` *before* extending with the step's own codes
         (exclusive: a step's action never leaks into its own gru);
      4. map the snapshot back to every row's ``video_id``.

    Robust to step gaps (accumulates over the present sorted steps only) and to
    duplicate steps (all rows at a step receive the identical gru).
    """
    # traj -> {step: own_codes}
    episodes: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
    for rec in records:
        traj, step = split_video_id(str(rec.get("video_id", "")))
        # unique step: identical duplicates are harmless; if codes ever differ
        # keep the first seen (deterministic w.r.t. input order).
        if step not in episodes[traj]:
            episodes[traj][step] = action_codes_from_answer(str(rec.get("a", "")))

    # traj -> {step: gru}
    step_gru: Dict[str, Dict[int, List[int]]] = {}
    for traj, step_codes in episodes.items():
        running: List[int] = []
        per_step: Dict[int, List[int]] = {}
        for step in sorted(step_codes):
            per_step[step] = list(running) + [STOP]
            running.extend(step_codes[step])
        step_gru[traj] = per_step

    # assign back to every row (including duplicates) by video_id
    out: Dict[str, List[int]] = {}
    for rec in records:
        vid = str(rec.get("video_id", ""))
        traj, step = split_video_id(vid)
        out[vid] = step_gru[traj][step]
    return out

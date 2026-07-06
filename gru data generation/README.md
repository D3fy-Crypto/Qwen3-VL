# GRU data generation

Scripts that add a per-record **`gru`** action-prefix field to the navigation
annotations so the GRU-Qwen SFT pipeline can inject a trajectory vector per sample.

## What `gru` is

The GRU-Qwen model encodes, for each nav sample, the **history of actions taken so
far** in the episode. That history is stored per row as an integer action-code list:

```
STOP = 0   FORWARD = 1   TURN_LEFT = 2   TURN_RIGHT = 3

gru(step S) = concat( action_codes(a) for every prior step k < S ) + [0]
```

- `action_codes(a)` parses the answer text: `turn right 45 degree → [3,3,3]`,
  `move forward 75 cm → [1,1,1]`, `stop → []` (see `nav_action_encoding.py`).
- The trailing **`0` is a structural placeholder for the *current* node** (the action
  still to be predicted), **not** a STOP the agent emitted. It is on every row. A real
  STOP answer parses to `[]`, so genuine stops never add an interior `0` — the only `0`
  is the tail.
- **`gru` length is unrelated to `len(frames)`.** `gru` is the history of actions so far
  (each turn/forward expands to several codes); `frames` is however many frames a slice
  stores. They are independent and must not be compared. They coincide for the simulator
  datasets R2R/RxR (one frame per motion primitive) but **not** for real-video Human.
- The scheme is **exclusive**: a step's own action never appears in its own `gru` (no
  label leakage); it first shows up in the next step's prefix.

Worked example — R2R episode `"1"` (matches the on-disk file exactly):

| video_id | answer `a` | `gru` |
|---|---|---|
| `1-0` | turn right 45° | `[0]` |
| `1-1` | turn right 45° | `[3,3,3,0]` |
| `1-2` | move forward 75 cm | `[3,3,3,3,3,3,0]` |
| `1-3` | move forward 75 cm | `[3,3,3,3,3,3,1,1,1,0]` |

## How the full trajectory is reconstructed

The annotations have **no trajectory object** — each episode is exploded into one row
per node, `video_id = "<traj>-<step>"`, where the row's answer `a` is the action at that
node. We invert the explosion: group rows by trajectory (`rsplit("-",1)`), order by
step, and accumulate each step's parsed action into a running prefix (snapshotting
`prefix + [0]` *before* extending — that's what keeps it exclusive). Robust to step
**gaps** (accumulate over present steps only) and **duplicate** rows (e.g. Human's 5×
stop-step → all get the same `gru`).

## Datasets

| Dataset | Status |
|---|---|
| **RxR**  | generated here (`{video_id,q,a,frames}`) |
| **Human**| generated here (has step gaps + oversampled stop-steps, handled) |
| **R2R**  | already on disk (`R2R/annotations_with_gru.json`); regenerate only to **verify** (must match exactly) |
| **EnvDrop** | NOT handled — `EnvDrop/envdrop_motion.json` already has an inline per-frame `motion` list (a different description-task scheme) |

## Usage

```bash
cd "gru data generation"

# Verify the algorithm reproduces the existing R2R file exactly (0 diffs):
python verify_gru_annotations.py --dataset r2r

# Generate the missing files (written to ./generated/, git-ignored):
python add_gru_to_annotations.py --dataset rxr
python add_gru_to_annotations.py --dataset human

# Validate what was generated:
python verify_gru_annotations.py --dataset rxr   --generated generated/rxr_annotations_with_gru.json
python verify_gru_annotations.py --dataset human --generated generated/human_annotations_with_gru.json

# Quick sample run (first N rows) for spot checks:
python add_gru_to_annotations.py --dataset human --limit 5000
```

The NaVILA-Dataset root is auto-detected (`/opt/IROS_proj/...` then
`/home/rithvik/IROS_proj/...`); override with `--navila-base`.

## Verification layers (`verify_gru_annotations.py`)

1. **Structural** (all): every row has `gru`; ends in `0`; codes ∈ {0,1,2,3}; `0` only
   at the tail.
2. **Oracle** (R2R, RxR): each step's parsed codes equal the colleague's
   `action_dict_all.json` body.
3. **Anchor** (R2R): recomputed `gru` matches on-disk `R2R/annotations_with_gru.json`
   exactly — this is the correctness proof that transfers to RxR/Human (identical schema
   and rule).

## Notes

- Outputs live in `generated/` and are **git-ignored** — they are large and meant for
  local verification, not for committing. To deploy, copy the validated file into the
  dataset dir (e.g. `RxR/annotations_with_gru.json`) and register an `rxr_gru`/`human_gru`
  entry in `qwen-vl-finetune/qwenvl/data/__init__.py` (separate follow-up).
- Supersedes the buggy `qwen-vl-finetune/tools/convert_r2r_add_gru_actions.py` (inclusive
  scheme → leaks the answer; does not reproduce the on-disk R2R file).
- `nav_action_encoding.py` copies `action_codes_from_answer` / `split_video_id` verbatim
  from `qwen-vl-finetune/qwenvl/data/data_processor_gru_chang.py`; the anchor check pins
  them in sync.

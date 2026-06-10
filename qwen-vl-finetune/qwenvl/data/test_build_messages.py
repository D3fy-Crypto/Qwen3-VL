"""
Smoke-test _build_messages on every dataset used by a training script.

It reads the DATASETS="..." line straight out of the slurm training script
(default: scripts/slurm_sft.sh), so the test always tracks whatever the script
is actually training on. For each dataset it loads the first NUM_SAMPLES
annotation items, runs _build_messages, and writes the full output (message
structure, image counts, has_missing, and frame counts for navigation samples)
to BOTH the console and a log file.

Usage:
    python qwen-vl-finetune/qwenvl/data/test_build_messages.py [LOG_PATH]

No model / GPU required — only _build_messages is exercised.
"""

import datetime
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwenvl.data import data_dict, parse_sampling_rate
from qwenvl.data.data_processor import _build_messages

# ---- Config -------------------------------------------------------------
SLURM_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "slurm_sft.sh"
NUM_SAMPLES = 3  # annotation items to test per dataset
DEFAULT_LOG = Path(__file__).resolve().parent / "test_build_messages.log"

# Model for the apply_chat_template check. Resolved in order: $MODEL_PATH env,
# the MODEL_PATH=... line in the slurm script, then these local fallbacks so the
# check can run offline. If none load, the chat-template section is skipped.
MODEL_PATH_FALLBACKS = [
    "/opt/IROS_proj/models/final_checkpoint",
]


class Tee:
    """Fan-out writes to several streams (console + log file).

    Resilient: if one stream errors (e.g. a broken console pipe when piped to
    `head`), the others keep working so the log file stays complete.
    """

    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except (BrokenPipeError, ValueError, OSError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass


def parse_datasets_from_slurm(script_path: Path):
    """Pull the comma-separated DATASETS="..." list out of a slurm script."""
    if not script_path.exists():
        print(f"[WARN] slurm script not found: {script_path}")
        print("[WARN] falling back to all datasets in data_dict")
        return list(data_dict.keys())
    text = script_path.read_text()
    match = re.search(r'^\s*DATASETS=["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        print(f"[WARN] no DATASETS=... line found in {script_path}")
        print("[WARN] falling back to all datasets in data_dict")
        return list(data_dict.keys())
    return [d.strip() for d in match.group(1).split(",") if d.strip()]


def iter_items(path: str, n: int):
    """Yield up to n individual annotation dicts from a .json/.jsonl file.

    Unwraps grouped list-entries (the data-packing format) into their
    sub-items so every yielded value is a single sample dict.
    """

    def raw_entries():
        p = Path(path)
        if p.suffix == ".jsonl":
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return
        # .json — stream with ijson to avoid loading multi-GB files fully.
        try:
            import ijson

            with open(p, "rb") as f:
                yield from ijson.items(f, "item")
        except ImportError:
            print("  [WARN] ijson not installed; loading whole JSON into memory")
            with open(p, "r") as f:
                yield from json.load(f)

    count = 0
    for entry in raw_entries():
        for item in (entry if isinstance(entry, list) else [entry]):
            if count >= n:
                return
            yield item
            count += 1


def preview_item(item) -> str:
    """Compact structural view of an item without dumping heavy fields."""
    if not isinstance(item, dict):
        return f"<non-dict item: {type(item).__name__}>"
    info = {"keys": list(item.keys())}
    if "frames" in item:
        info["num_frames"] = len(item["frames"])
    if "image" in item:
        img = item["image"]
        info["num_images"] = len(img) if isinstance(img, list) else 1
    if "video" in item:
        info["video"] = item["video"]
    if "video_id" in item:
        info["video_id"] = item["video_id"]
    return json.dumps(info, ensure_ascii=False)


def count_images(messages) -> int:
    return sum(
        1
        for m in messages
        for c in m["content"]
        if isinstance(c, dict) and c.get("type") == "image"
    )


def summarize_messages(messages) -> str:
    import PIL.Image

    def serialize(obj):
        if isinstance(obj, PIL.Image.Image):
            return f"<PIL Image mode={obj.mode} size={obj.size}>"
        raise TypeError(type(obj))

    return json.dumps(messages, indent=2, ensure_ascii=False, default=serialize)


def parse_model_path_from_slurm(script_path: Path):
    """Pull the MODEL_PATH="..." value out of a slurm script, or None."""
    if not script_path.exists():
        return None
    match = re.search(
        r'^\s*MODEL_PATH=["\']([^"\']+)["\']', script_path.read_text(), re.MULTILINE
    )
    return match.group(1) if match else None


def resolve_model_path():
    """First existing of: $MODEL_PATH env, slurm MODEL_PATH, local fallbacks."""
    import os

    candidates = [
        os.environ.get("MODEL_PATH"),
        parse_model_path_from_slurm(SLURM_SCRIPT),
        *MODEL_PATH_FALLBACKS,
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def load_processor():
    """Load the processor for apply_chat_template; return None if unavailable."""
    model_path = resolve_model_path()
    if model_path is None:
        print("[WARN] no usable model path (env MODEL_PATH / slurm / fallbacks);")
        print("[WARN] skipping apply_chat_template section")
        return None
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"Processor   : {type(proc).__name__} <- {model_path}")
        return proc
    except Exception as e:
        print(f"[WARN] failed to load processor from {model_path}: {e}")
        print("[WARN] skipping apply_chat_template section")
        return None


def show_chat_template(processor, messages):
    """Render the built messages through apply_chat_template (mirrors training)."""
    print("  --- apply_chat_template (tokenize=False) ---")
    try:
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        print(f"  [WARN] apply_chat_template(tokenize=False) failed: {e}")
        return
    print(f"  has_system_block={'<|im_start|>system' in rendered}")
    print(rendered)
    # Tokenized pass mirrors preprocess_qwen_visual. Note: default pixel range here,
    # so input_ids_len differs from training (which sets min/max_pixels first).
    try:
        out = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        info = {"input_ids_len": int(out["input_ids"].shape[-1])}
        if "image_grid_thw" in out:
            info["image_grid_thw"] = out["image_grid_thw"].tolist()
        if "pixel_values" in out:
            info["pixel_values_shape"] = list(out["pixel_values"].shape)
        print(f"  tokenized (default pixels): {json.dumps(info)}")
    except Exception as e:
        print(f"  [WARN] apply_chat_template(tokenize=True) failed: {e}")


def test_one_item(item, data_path: Path, processor=None):
    print(f"  raw item: {preview_item(item)}")
    if not isinstance(item, dict):
        print("  [FAIL] item is not a dict, cannot build messages")
        return
    try:
        messages, has_missing = _build_messages(item, data_path)
    except Exception as e:
        print(f"  [FAIL] _build_messages: {e}")
        import traceback

        traceback.print_exc()
        return
    print(
        f"  [OK] _build_messages: {len(messages)} turns, "
        f"{count_images(messages)} images, has_missing={has_missing}"
    )
    print(summarize_messages(messages))
    if processor is not None:
        show_chat_template(processor, messages)


def test_dataset(display_name: str, cfg: dict, sampling_rate: float, processor=None):
    print("\n" + "=" * 70)
    print(f"Dataset: {display_name}  (sampling_rate={sampling_rate})")
    print(f"  annotation: {cfg['annotation_path']}")
    print(f"  data_path:  {cfg['data_path']}")

    ann_path = Path(cfg["annotation_path"])
    if not ann_path.exists():
        print("  [SKIP] annotation file not found")
        return
    if not Path(cfg["data_path"]).exists():
        print("  [WARN] data_path does not exist — images will load as black frames")

    try:
        items = list(iter_items(str(ann_path), NUM_SAMPLES))
    except Exception as e:
        print(f"  [FAIL] reading annotations: {e}")
        import traceback

        traceback.print_exc()
        return

    if not items:
        print("  [SKIP] no items in annotation file")
        return

    data_path = Path(cfg["data_path"])
    for idx, item in enumerate(items):
        print(f"\n  --- sample #{idx} ---")
        test_one_item(item, data_path, processor)


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # line-buffered so the log can be tailed live
    log_file = open(log_path, "w", buffering=1)
    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee
    sys.stderr = tee
    # Route data_processor's "Missing image" warnings into the same log.
    logging.basicConfig(stream=tee, level=logging.WARNING, force=True)

    try:
        print(f"Log started : {datetime.datetime.now().isoformat()}")
        print(f"Slurm script: {SLURM_SCRIPT}")
        print(f"Samples/ds  : {NUM_SAMPLES}")
        print(f"Log file    : {log_path}")

        dataset_names = parse_datasets_from_slurm(SLURM_SCRIPT)
        print(f"Datasets    : {dataset_names}")

        processor = load_processor()

        for ds in dataset_names:
            sampling_rate = parse_sampling_rate(ds)
            raw_name = re.sub(r"%(\d+)$", "", ds)
            cfg = data_dict.get(raw_name)
            if cfg is None:
                print("\n" + "=" * 70)
                print(f"Dataset: {ds}")
                print("  [FAIL] unknown dataset name (not in data_dict)")
                continue
            test_dataset(ds, cfg, sampling_rate, processor)

        print("\n" + "=" * 70)
        print("All datasets tested.")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log written to: {log_path}")


if __name__ == "__main__":
    main()

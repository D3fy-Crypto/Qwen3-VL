"""
Quick test: run _build_messages + preprocess_qwen_visual on the first item
of R2R_ALIGNMENT_QA and show messages, decoded input_ids, and labels.
Usage: python qwen-vl-finetune/qwenvl/data/test_build_messages.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwenvl.data.data_processor import _build_messages, preprocess_qwen_visual

ANNOTATION_PATH = "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset/R2R/annotations.json"
DATA_PATH       = "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset/R2R/train"
MODEL_PATH      = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Model/instruct"


def load_first_item(path: str) -> dict:
    try:
        import ijson
        with open(path, "rb") as f:
            return next(ijson.items(f, "item"))
    except ImportError:
        with open(path, "rb") as f:
            f.read(1)
            buf, depth = b"", 0
            while True:
                ch = f.read(1)
                if not ch:
                    break
                buf += ch
                if ch in (b"{", b"["):
                    depth += 1
                elif ch in (b"}", b"]"):
                    depth -= 1
                    if depth == 0:
                        break
        return json.loads(buf)


def main():
    print(f"Loading first item from: {ANNOTATION_PATH}")
    item = load_first_item(ANNOTATION_PATH)
    item["data_path"] = DATA_PATH

    print("\n=== Raw dataset item ===")
    print(json.dumps(item, indent=2, ensure_ascii=False))

    # ── Step 1: _build_messages ──────────────────────────────────────────────
    messages = _build_messages(item, Path(DATA_PATH))

    print("\n=== Messages before apply_chat_template ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))

    print("\n=== Summary ===")
    for i, msg in enumerate(messages):
        if isinstance(msg["content"], str):
            content_types = ["text"]
            preview = msg["content"][:80]
        else:
            content_types = [c["type"] for c in msg["content"]]
            preview = " | ".join(
                str(c.get("text", c.get("image", c.get("video", ""))))[:40]
                for c in msg["content"]
            )
        print(f"  [{i}] role={msg['role']:<10} types={content_types}")
        print(f"       preview: {preview}")

    # ── Step 2: preprocess_qwen_visual (requires loading processor) ──────────
    print(f"\nLoading processor from: {MODEL_PATH}")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    data_dict = preprocess_qwen_visual([item], processor)

    tokenizer = processor.tokenizer
    IGNORE_INDEX = -100

    decoded_input = tokenizer.decode(data_dict["input_ids"][0], skip_special_tokens=False)
    print("\n=== Decoded input_ids (full prompt seen by model) ===")
    print(decoded_input)

    label_ids = [
        tid if tid != IGNORE_INDEX else tokenizer.pad_token_id
        for tid in data_dict["labels"][0].tolist()
    ]
    decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False)
    print("\n=== Decoded labels (what the model is trained to predict) ===")
    print(decoded_labels)

    n_label_tokens = sum(1 for t in data_dict["labels"][0].tolist() if t != IGNORE_INDEX)
    print(f"\nTotal tokens: {data_dict['input_ids'].shape[1]}  |  Label tokens: {n_label_tokens}")


if __name__ == "__main__":
    main()

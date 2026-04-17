"""
Quick test: run _build_messages on the first item of R2R_ALIGNMENT_QA.
Usage: python qwen-vl-finetune/qwenvl/data/test_build_messages.py
"""

import json
import sys
from pathlib import Path

# Make sure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwenvl.data.data_processor import _build_messages

ANNOTATION_PATH = "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset/R2R/no_gru_R2R.json"
DATA_PATH       = "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset/R2R/train"

def main():
    print(f"Loading first item from: {ANNOTATION_PATH}")
    try:
        import ijson
        with open(ANNOTATION_PATH, "rb") as f:
            item = next(ijson.items(f, "item"))
    except ImportError:
        # fallback: read raw bytes until we close the first JSON object
        with open(ANNOTATION_PATH, "rb") as f:
            f.read(1)  # skip opening '['
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
        item = json.loads(buf)
    item["data_path"] = DATA_PATH
    base_path = Path(DATA_PATH)

    print("\n=== Raw dataset item ===")
    print(json.dumps(item, indent=2, ensure_ascii=False))

    print("\n=== _build_messages output ===")
    messages = _build_messages(item, base_path)
    print(json.dumps(messages, indent=2, ensure_ascii=False))

    print(f"\nTotal turns: {len(messages)}")
    for i, msg in enumerate(messages):
        content_types = [c["type"] for c in msg["content"]]
        print(f"  turn {i}: role={msg['role']}, content_types={content_types}")

if __name__ == "__main__":
    main()

"""
Test _build_messages on the first item of every dataset.
Usage: python qwen-vl-finetune/qwenvl/data/test_build_messages.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwenvl.data.data_processor import _build_messages
from qwenvl.data import data_dict as DATASETS

MODEL_PATH = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Model/instruct"


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


def summarize_messages(messages):
    import PIL.Image

    def serialize(obj):
        if isinstance(obj, PIL.Image.Image):
            return f"<PIL Image mode={obj.mode} size={obj.size}>"
        raise TypeError(type(obj))

    print(json.dumps(messages, indent=2, ensure_ascii=False, default=serialize))


def test_dataset(name, cfg):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  annotation: {cfg['annotation_path']}")
    print(f"  data_path:  {cfg['data_path']}")

    ann_path = Path(cfg["annotation_path"])
    if not ann_path.exists():
        print(f"  [SKIP] annotation file not found")
        return

    try:
        item = load_first_item(str(ann_path))
    except Exception as e:
        print(f"  [FAIL] load_first_item: {e}")
        return

    data_path = Path(cfg["data_path"])
    try:
        messages = _build_messages(item, data_path)
        print(f"  [OK] _build_messages succeeded, {len(messages)} messages")
        summarize_messages(messages)
    except Exception as e:
        print(f"  [FAIL] _build_messages: {e}")
        import traceback; traceback.print_exc()


def main():
    for name, cfg in DATASETS.items():
        test_dataset(name, cfg)

    print(f"\n{'='*60}")
    print("All datasets tested.")


if __name__ == "__main__":
    main()

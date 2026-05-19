"""
Sample 10 items from each of scanqa, r2r, rxr, human datasets plus 10 MMMU
questions, run inference on both base (port 8000) and SFT (port 8001) servers,
and save question/ground-truth/response logs to inference/logs/.

Usage:
    python inference/run_eval.py
    python inference/run_eval.py --n 5 --base-port 8000 --sft-port 8001
    python inference/run_eval.py --datasets r2r scanqa mmmu
    python inference/run_eval.py --no-mmmu   # skip MMMU section
"""

import argparse
import base64
import io
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import requests
from PIL import Image as PILImage

# ── Dataset configs (mirrors qwenvl/data/__init__.py) ──────────────────────────
DATASET_CONFIGS = {
    "r2r": {
        "annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/annotations.json",
        "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/train",
    },
    "rxr": {
        "annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/RxR/annotations.json",
        "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/RxR/train",
    },
    "human": {
        "annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/Human/annotations.json",
        "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/Human/raw_frames",
    },
    "scanqa": {
        "annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
        "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/ScanQA/videos",
    },
}

NUM_HISTORICAL_FRAMES = 7
SYSTEM_PROMPT = "You are a helpful navigation assistant."
MMMU_SUBJECTS = [
    "Art", "Art_Theory", "Biology", "Chemistry", "Clinical_Medicine",
    "Computer_Science", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing",
    "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

LOG_DIR = Path(__file__).parent / "logs"


# ── Image utilities ─────────────────────────────────────────────────────────────

def pil_to_b64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def path_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{data}"


def image_to_server_value(img) -> str:
    """Convert PIL Image or file path string to a value the server accepts."""
    if isinstance(img, PILImage.Image):
        return pil_to_b64(img)
    return path_to_b64(str(img))  # file path → base64


def extract_video_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
    indices = [int(i) for i in __import__("numpy").linspace(0, total - 1, min(total, num_frames))]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        elif frames:
            frames.append(frames[-1])
        else:
            frames.append(PILImage.new("RGB", (224, 224)))
    cap.release()
    return frames


# ── Message builders ────────────────────────────────────────────────────────────

def build_scanqa_messages(item: dict, data_path: str) -> tuple[list, str]:
    """Returns (messages_without_assistant, ground_truth)."""
    video_path = str(Path(data_path) / f"{item['video_id']}.mp4")
    frames = extract_video_frames(video_path, NUM_HISTORICAL_FRAMES + 1)
    gt = random.choice(item["a"]) if isinstance(item["a"], list) else item["a"]
    content = [
        *[{"type": "image", "image": image_to_server_value(f)} for f in frames],
        {"type": "text", "text": item["q"]},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, gt


def build_nav_messages(item: dict, data_path: str) -> tuple[list, str]:
    """Returns (messages_without_assistant, ground_truth). For R2R / RxR / Human."""
    base = Path(data_path)
    frames = item["frames"]
    current_path = str(base / frames[-1])
    historical_pool = frames[:-1]

    import numpy as np
    n = len(historical_pool)
    if n > 0:
        indices = np.linspace(0, n - 1, min(n, NUM_HISTORICAL_FRAMES), dtype=int).tolist()
        historical_paths = [str(base / historical_pool[i]) for i in indices]
    else:
        historical_paths = []

    pad_count = NUM_HISTORICAL_FRAMES - len(historical_paths)
    black = PILImage.new("RGB", (224, 224))
    loaded_historical = [pil_to_b64(black)] * pad_count + [
        path_to_b64(p) if Path(p).exists() else pil_to_b64(black)
        for p in historical_paths
    ]
    current_val = path_to_b64(current_path) if Path(current_path).exists() else pil_to_b64(black)

    content = [
        {"type": "text", "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations:"},
        *[{"type": "image", "image": v} for v in loaded_historical],
        {"type": "text", "text": "and current observation:"},
        {"type": "image", "image": current_val},
        {"type": "text", "text": (
            f"Your assigned task is: {item['q']}\n"
            "Analyze this series of images to decide your next move, which could involve "
            "turning left or right by a specific degree, moving forward a certain distance, "
            "or stop if the task is completed."
        )},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, item["a"]


# ── MMMU ───────────────────────────────────────────────────────────────────────

def load_mmmu_items(n: int, seed: int = 42) -> list:
    """Load n MMMU validation items, sampling across subjects."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install huggingface datasets: pip install datasets")

    random.seed(seed)
    subjects = random.sample(MMMU_SUBJECTS, min(len(MMMU_SUBJECTS), n))
    items = []
    for subject in subjects:
        try:
            ds = load_dataset("MMMU/MMMU", subject, split="validation", trust_remote_code=True)
            row = ds[random.randrange(len(ds))]
            items.append({"subject": subject, "row": row})
            if len(items) >= n:
                break
        except Exception as e:
            print(f"  [MMMU] skipping {subject}: {e}")
    return items[:n]


def build_mmmu_messages(item: dict) -> tuple[list, str]:
    """Format an MMMU item as server messages. Returns (messages, ground_truth)."""
    row = item["row"]
    options = row.get("options", [])
    option_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
    question_text = row["question"].replace("<image 1>", "").replace("<image 2>", "").strip()
    prompt = f"{question_text}\n\nOptions:\n{option_str}\n\nAnswer with the option letter only (A, B, C, or D)."

    content = []
    for k in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
        img = row.get(k)
        if img is not None:
            content.append({"type": "image", "image": pil_to_b64(img.convert("RGB"))})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    gt = row.get("answer", "")
    return messages, gt


# ── Server calls ────────────────────────────────────────────────────────────────

def call_server(base_url: str, messages: list, max_new_tokens: int = 256) -> str:
    try:
        resp = requests.post(
            f"{base_url}/generate",
            json={"messages": messages, "max_new_tokens": max_new_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f"[ERROR] {e}"


def check_server(base_url: str, name: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        info = resp.json()
        print(f"  [{name}] {base_url} — {info.get('model', '?')}")
        return True
    except Exception as e:
        print(f"  [{name}] UNREACHABLE — {e}")
        return False


# ── Main ────────────────────────────────────────────────────────────────────────

def load_n_items(annotation_path: str, n: int, seed: int = 42) -> list:
    print(f"  Loading {annotation_path} ...")
    with open(annotation_path) as f:
        data = json.load(f)
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_eval(args):
    base_url = f"http://{args.host}:{args.base_port}"
    sft_url = f"http://{args.host}:{args.sft_port}"

    print("\n=== Checking servers ===")
    ok_base = check_server(base_url, "BASE")
    ok_sft = check_server(sft_url, "SFT")
    if not ok_base or not ok_sft:
        print("Both servers must be running. Exiting.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f"eval_{timestamp}.json"
    txt_path = LOG_DIR / f"eval_{timestamp}.txt"

    all_results = []

    for ds_name in args.datasets:
        cfg = DATASET_CONFIGS[ds_name]
        print(f"\n=== {ds_name.upper()} ===")
        items = load_n_items(cfg["annotation_path"], args.n)

        for i, item in enumerate(items):
            print(f"  [{i+1}/{args.n}] {item.get('video_id', '')} ...", end=" ", flush=True)
            try:
                if ds_name == "scanqa":
                    messages, gt = build_scanqa_messages(item, cfg["data_path"])
                else:
                    messages, gt = build_nav_messages(item, cfg["data_path"])

                base_resp = call_server(base_url, messages, args.max_new_tokens)
                sft_resp = call_server(sft_url, messages, args.max_new_tokens)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                base_resp = f"[BUILD ERROR] {e}"
                sft_resp = f"[BUILD ERROR] {e}"
                gt = item.get("a", "")
                if isinstance(gt, list):
                    gt = gt[0]

            entry = {
                "dataset": ds_name,
                "index": i,
                "video_id": item.get("video_id", ""),
                "question": item["q"],
                "ground_truth": gt if isinstance(gt, str) else gt[0],
                "base_response": base_resp,
                "sft_response": sft_resp,
            }
            all_results.append(entry)

    # ── MMMU section ──
    if not args.no_mmmu:
        print(f"\n=== MMMU (general benchmark, {args.n} questions) ===")
        try:
            mmmu_items = load_mmmu_items(args.n, seed=args.seed)
            for i, item in enumerate(mmmu_items):
                subject = item["subject"]
                print(f"  [{i+1}/{len(mmmu_items)}] {subject} ...", end=" ", flush=True)
                try:
                    messages, gt = build_mmmu_messages(item)
                    base_resp = call_server(base_url, messages, max_new_tokens=16)
                    sft_resp = call_server(sft_url, messages, max_new_tokens=16)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")
                    base_resp = sft_resp = f"[BUILD ERROR] {e}"
                    gt = item["row"].get("answer", "")

                row = item["row"]
                options = row.get("options", [])
                option_str = " / ".join(f"{chr(65+j)}) {opt}" for j, opt in enumerate(options))
                all_results.append({
                    "dataset": "mmmu",
                    "index": i,
                    "video_id": f"{subject}/{row.get('id', '')}",
                    "question": row["question"],
                    "options": option_str,
                    "ground_truth": gt,
                    "base_response": base_resp,
                    "sft_response": sft_resp,
                })
        except Exception as e:
            print(f"  [MMMU] Failed to load: {e}")

    # ── Save JSON ──
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Save readable text ──
    with open(txt_path, "w") as f:
        f.write(f"Eval run: {timestamp}\n")
        f.write(f"Base server: {base_url}\n")
        f.write(f"SFT  server: {sft_url}\n")
        f.write("=" * 80 + "\n\n")
        for r in all_results:
            f.write(f"[{r['dataset'].upper()}] #{r['index']+1}  video_id={r['video_id']}\n")
            f.write(f"Q:    {r['question']}\n")
            if "options" in r:
                f.write(f"Opts: {r['options']}\n")
            f.write(f"GT:   {r['ground_truth']}\n")
            f.write(f"BASE: {r['base_response']}\n")
            f.write(f"SFT:  {r['sft_response']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"\n=== Done ===")
    print(f"  JSON log : {json_path}")
    print(f"  Text log : {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--sft-port", type=int, default=8001)
    parser.add_argument("--n", type=int, default=10, help="Items per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--datasets", nargs="+", default=["r2r", "rxr", "human", "scanqa"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mmmu", action="store_true", help="Skip MMMU general benchmark questions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)

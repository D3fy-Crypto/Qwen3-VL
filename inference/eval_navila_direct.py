"""
Standalone NaVILA eval — no server needed.
Loads the NaVILA model directly and runs inference on 10 items from each of
r2r, rxr, human, scanqa, plus 10 MMMU questions. Saves logs to inference/logs/.

Must be run in an environment compatible with NaVILA's llava package
(transformers ~4.37, NOT the qwen-eval env which has transformers 5.x).

Usage:
    conda activate <navila-env>
    cd /home/chang/Projects/vla_proj/Qwen3-VL
    python inference/eval_navila_direct.py
    python inference/eval_navila_direct.py --n 5 --no-mmmu
"""

import argparse
import base64
import io
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# ── NaVILA repo on path ──────────────────────────────────────────────────────
NAVILA_REPO = "/home/rithvik/IROS_proj/cvpr_proj/NaVILA"
if NAVILA_REPO not in sys.path:
    sys.path.insert(0, NAVILA_REPO)

# ── Block deepspeed import ────────────────────────────────────────────────────
# llava.model imports llava.train.utils → llava.train.sequence_parallel.globals
# → deepspeed.comm. deepspeed is broken in both qwen-eval and navila-eval
# environments. Stub out just this one module to cut the chain.
import types as _types

_sp_globals = _types.ModuleType("llava.train.sequence_parallel.globals")
_sp_globals.get_pg_manager = lambda: None
_sp_globals.set_pg_manager = lambda *a, **kw: None
_sp_globals.get_ulysses_sp_pg = lambda: None
_sp_globals.get_data_parallel_rank = lambda: 0
_sp_globals.get_sequence_parallel_rank = lambda: 0
_sp_globals.get_sequence_parallel_world_size = lambda: 1

_sp_pkg = _types.ModuleType("llava.train.sequence_parallel")
_sp_pkg.globals = _sp_globals
_sp_pkg.get_pg_manager = _sp_globals.get_pg_manager
_sp_pkg.set_pg_manager = _sp_globals.set_pg_manager

sys.modules["llava.train.sequence_parallel"] = _sp_pkg
sys.modules["llava.train.sequence_parallel.globals"] = _sp_globals

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/rithvik/IROS_proj/NaVILA_Pretrained/navila-llama3-8b-8f"
MODEL_NAME  = "navila-llama3-8b-8f"
CONV_MODE   = "llama_3"
NUM_FRAMES  = 8  # 7 historical + 1 current

DATASET_CONFIGS = {
    "r2r":    {"annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/annotations.json",
               "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/R2R/train"},
    "rxr":    {"annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/RxR/annotations.json",
               "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/RxR/train"},
    "human":  {"annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/Human/annotations.json",
               "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/Human/raw_frames"},
    "scanqa": {"annotation_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
               "data_path": "/home/rithvik/IROS_proj/NaVILA-Dataset/ScanQA/videos"},
}

MMMU_SUBJECTS = [
    "Art", "Art_Theory", "Biology", "Chemistry", "Clinical_Medicine",
    "Computer_Science", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing",
    "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

NAV_PROMPT = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and a current observation. "
    "Your assigned task is: {instruction}\n"
    "Analyze this series of observations to decide your next action, which could be "
    "turning left or right by a specific degree, moving forward a certain distance, "
    "or stop if the task is completed."
)

LOG_DIR = Path(__file__).parent / "logs"


# ── Image utilities ──────────────────────────────────────────────────────────

def extract_video_frames(video_path: str, num_frames: int) -> list[PILImage.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, total - 1, min(total, num_frames), dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret
                      else (frames[-1] if frames else PILImage.new("RGB", (224, 224))))
    cap.release()
    return frames


def load_frame(path: str) -> PILImage.Image:
    p = Path(path)
    return PILImage.open(p).convert("RGB") if p.exists() else PILImage.new("RGB", (224, 224))


def pad_frames(frames: list, target: int) -> list:
    black = PILImage.new("RGB", (224, 224))
    while len(frames) < target:
        frames.insert(0, black)
    return frames


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, image_processor, images: list[PILImage.Image],
                  text: str, max_new_tokens: int = 256) -> str:
    images = pad_frames(images, NUM_FRAMES)
    prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + text
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Stop token matches run_vila.py pattern
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer,
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0))

    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0).to(model.device)
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=[images_tensor],
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # LlavaLlama.generate() returns only new tokens (not input prefix).
    # Decode directly without slicing — same pattern as run_vila.py.
    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if stop_str and out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out


# ── Dataset builders ─────────────────────────────────────────────────────────

def build_scanqa(item: dict, data_path: str):
    frames = extract_video_frames(str(Path(data_path) / f"{item['video_id']}.mp4"), NUM_FRAMES)
    gt = random.choice(item["a"]) if isinstance(item["a"], list) else item["a"]
    return frames, item["q"], gt


def build_nav(item: dict, data_path: str):
    base = Path(data_path)
    frame_paths = item["frames"]
    current = load_frame(str(base / frame_paths[-1]))
    pool = frame_paths[:-1]
    if pool:
        indices = np.linspace(0, len(pool) - 1, min(len(pool), NUM_FRAMES - 1), dtype=int).tolist()
        hist = [load_frame(str(base / pool[i])) for i in indices]
    else:
        hist = []
    frames = pad_frames(hist, NUM_FRAMES - 1) + [current]
    return frames, NAV_PROMPT.format(instruction=item["q"]), item["a"]


# ── MMMU ─────────────────────────────────────────────────────────────────────

def cached_mmmu_subjects() -> list[str]:
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "MMMU___mmmu"
    if not cache_dir.exists():
        return []
    return [d.name for d in sorted(cache_dir.iterdir()) if d.is_dir() and d.name in MMMU_SUBJECTS]


def load_mmmu_items(n: int, seed: int) -> list:
    from datasets import load_dataset
    cached = cached_mmmu_subjects()
    if not cached:
        raise RuntimeError("No MMMU subjects cached locally.")
    print(f"  [MMMU] cached: {cached}")
    random.seed(seed)
    pool = random.sample(cached, min(len(cached), n))
    if len(pool) < n:
        pool = (pool * (n // len(pool) + 1))[:n]
    items = []
    for subject in pool:
        try:
            ds = load_dataset("MMMU/MMMU", subject, split="validation",
                              trust_remote_code=True,
                              download_mode="reuse_dataset_if_exists")
            items.append({"subject": subject, "row": ds[random.randrange(len(ds))]})
        except Exception as e:
            print(f"  [MMMU] skip {subject}: {e}")
    return items[:n]


def build_mmmu(item: dict):
    row = item["row"]
    options = row.get("options", [])
    option_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
    q = row["question"].replace("<image 1>", "").replace("<image 2>", "").strip()
    text = f"{q}\n\nOptions:\n{option_str}\n\nAnswer with the option letter only (A, B, C, or D)."
    images = [row[k].convert("RGB") for k in
              ["image_1","image_2","image_3","image_4","image_5","image_6","image_7"]
              if row.get(k) is not None]
    return images, text, row.get("answer", "")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    print(f"\nLoading NaVILA model from {MODEL_PATH} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=MODEL_PATH, model_name=MODEL_NAME, model_base=None, device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f"eval_navila_{timestamp}.json"
    txt_path  = LOG_DIR / f"eval_navila_{timestamp}.txt"

    all_results = []

    # ── Navigation datasets ──
    for ds_name in args.datasets:
        cfg = DATASET_CONFIGS[ds_name]
        print(f"=== {ds_name.upper()} ===")
        with open(cfg["annotation_path"]) as f:
            data = json.load(f)
        random.seed(args.seed)
        items = random.sample(data, min(args.n, len(data)))

        for i, item in enumerate(items):
            print(f"  [{i+1}/{args.n}] {item.get('video_id','')} ...", end=" ", flush=True)
            try:
                if ds_name == "scanqa":
                    frames, text, gt = build_scanqa(item, cfg["data_path"])
                else:
                    frames, text, gt = build_nav(item, cfg["data_path"])
                response = run_inference(model, tokenizer, image_processor, frames, text, args.max_new_tokens)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                response = f"[ERROR] {e}"
                gt = (item.get("a", "") or "")
                if isinstance(gt, list): gt = gt[0]

            all_results.append({
                "dataset": ds_name, "index": i,
                "video_id": item.get("video_id", ""),
                "question": item["q"],
                "ground_truth": gt if isinstance(gt, str) else gt[0],
                "navila_response": response,
            })

    # ── MMMU ──
    if not args.no_mmmu:
        print(f"\n=== MMMU ({args.n} questions) ===")
        try:
            mmmu_items = load_mmmu_items(args.n, args.seed)
            for i, item in enumerate(mmmu_items):
                print(f"  [{i+1}/{len(mmmu_items)}] {item['subject']} ...", end=" ", flush=True)
                try:
                    frames, text, gt = build_mmmu(item)
                    response = run_inference(model, tokenizer, image_processor, frames, text, max_new_tokens=16)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")
                    response = f"[ERROR] {e}"
                    gt = item["row"].get("answer", "")

                row = item["row"]
                opts = " / ".join(f"{chr(65+j)}) {o}" for j, o in enumerate(row.get("options", [])))
                all_results.append({
                    "dataset": "mmmu", "index": i,
                    "video_id": f"{item['subject']}/{row.get('id','')}",
                    "question": row["question"], "options": opts,
                    "ground_truth": gt, "navila_response": response,
                })
        except Exception as e:
            print(f"  [MMMU] failed: {e}")

    # ── Save ──
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w") as f:
        f.write(f"NaVILA direct eval: {timestamp}\n")
        f.write(f"Model: {MODEL_PATH}\n{'='*80}\n\n")
        for r in all_results:
            f.write(f"[{r['dataset'].upper()}] #{r['index']+1}  {r['video_id']}\n")
            f.write(f"Q:       {r['question']}\n")
            if "options" in r:
                f.write(f"Opts:    {r['options']}\n")
            f.write(f"GT:      {r['ground_truth']}\n")
            f.write(f"NaVILA: {r['navila_response']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"\n=== Done ===")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--datasets", nargs="+", default=["r2r", "rxr", "human", "scanqa"],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mmmu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())

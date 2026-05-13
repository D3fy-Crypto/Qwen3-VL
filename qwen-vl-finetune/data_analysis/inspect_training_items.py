"""
inspect_training_items.py
Equivalent of inspect_training_items.ipynb — filter training data and
batch-test items against the vanilla Qwen server.

Edit the CONFIG block below, then run:
    python inspect_training_items.py
"""

import json
import re
import base64
import socket
import time
import random
from datetime import datetime
from io import BytesIO
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Filter (Cell 2)
FILTER_DATASET  = "R2R"    # "R2R" | "Human" | "RxR" | "ScanQA" | None (all)
FILTER_ACTION   = "FORWARD"   # "STOP" | "FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "OTHER" | None (all)
FILTER_VIDEO_ID = None     # e.g. "914-23" or None
FILTER_KEYWORD  = None     # substring in question text, e.g. "stairs" or None
MAX_PREVIEW     = 20       # rows shown in the filter preview table

# Batch test (Cell 5)
N_TEST          = 1000     #  1000 items need around 8min on vega
RANDOM_SEED     = 42       # None = different sample every run
TARGET_ACTIONS  = ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"]
SERVER_HOST     = "localhost"
SERVER_PORT     = 54321
SAVE_DIR        = Path(__file__).parent   # where to write the result JSON
# ─────────────────────────────────────────────────────────────────────────────

NAVILA_BASE = Path("/home/rithvik/IROS_proj/NaVILA-Dataset")

DATASETS = {
    "R2R":    {"ann": NAVILA_BASE / "R2R/annotations.json",
               "frames_root": NAVILA_BASE / "R2R/train"},
    "Human":  {"ann": NAVILA_BASE / "Human/annotations.json",
               "frames_root": NAVILA_BASE / "Human/raw_frames"},
    "RxR":    {"ann": NAVILA_BASE / "RxR/annotations.json",
               "frames_root": NAVILA_BASE / "RxR/train"},
    "ScanQA": {"ann": NAVILA_BASE / "ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
               "frames_root": NAVILA_BASE / "ScanQA/videos"},
}


def classify_action(answer):
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    a = (answer or "").lower()
    if "right"   in a: return "TURN_RIGHT"
    if "left"    in a: return "TURN_LEFT"
    if "forward" in a: return "FORWARD"
    if "stop"    in a: return "STOP"
    return "OTHER"


def encode_image_b64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def send_request(host, port, image_b64, query, timeout=30):
    payload = json.dumps({"image": image_b64, "query": query}).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(len(payload).to_bytes(8, "big"))
        s.sendall(payload)
        sz = b""
        while len(sz) < 8:
            sz += s.recv(8 - len(sz))
        data = b""
        size = int.from_bytes(sz, "big")
        while len(data) < size:
            data += s.recv(4096)
    return json.loads(data.decode())


def parse_pred(text):
    t = text.lower()
    if re.search(r"\bforward\b", t):                               return "FORWARD"
    if re.search(r"\bturn(?:ed)?\s+left\b|\bleft\s+turn\b", t):  return "TURN_LEFT"
    if re.search(r"\bturn(?:ed)?\s+right\b|\bright\s+turn\b", t): return "TURN_RIGHT"
    if re.search(r"\bstop\b", t):                                  return "STOP"
    if re.search(r"\bleft\b", t):  return "TURN_LEFT"
    if re.search(r"\bright\b", t): return "TURN_RIGHT"
    return None


def extract_value(text, action):
    """Extract numeric value (cm for FORWARD, degrees for turns). Returns None if not applicable."""
    if action not in ("FORWARD", "TURN_LEFT", "TURN_RIGHT"):
        return None
    t = (text or "").lower()
    if action == "FORWARD":
        m = re.search(r"(\d+)\s*cm", t)
    else:
        m = re.search(r"(\d+)\s*degree", t)
    return int(m.group(1)) if m else None


def load_datasets():
    all_data = {}
    for ds_name, cfg in DATASETS.items():
        print(f"Loading {ds_name}...", end=" ", flush=True)
        with open(cfg["ann"]) as f:
            raw = json.load(f)
        for item in raw:
            item["_dataset"]     = ds_name
            item["_frames_root"] = cfg["frames_root"]
            item["_action"]      = classify_action(item.get("a", ""))
        all_data[ds_name] = raw
        print(f"{len(raw):,} items")
    return all_data


def filter_items(all_data):
    pool = []
    for ds_name, items in all_data.items():
        if FILTER_DATASET and ds_name != FILTER_DATASET:
            continue
        pool.extend(items)

    filtered = []
    for item in pool:
        if FILTER_ACTION   and item["_action"] != FILTER_ACTION:                           continue
        if FILTER_VIDEO_ID and item.get("video_id") != FILTER_VIDEO_ID:                    continue
        if FILTER_KEYWORD  and FILTER_KEYWORD.lower() not in item.get("q", "").lower():    continue
        filtered.append(item)

    print(f"\nFound {len(filtered):,} matching items.")
    if filtered:
        print(f"\n{'#':<6}  {'Dataset':<8}  {'Action':<12}  {'video_id':<22}  Question (truncated)")
        print("-" * 100)
        for i, item in enumerate(filtered[:MAX_PREVIEW]):
            q_short = item.get("q", "")[:55].replace("\n", " ")
            print(f"{i:<6}  {item['_dataset']:<8}  {item['_action']:<12}  "
                  f"{str(item.get('video_id','')):<22}  {q_short}")
        if len(filtered) > MAX_PREVIEW:
            print(f"  ... and {len(filtered)-MAX_PREVIEW:,} more")
    return filtered


def batch_test(filtered):
    pool     = [it for it in filtered if it["_action"] in TARGET_ACTIONS]
    rng      = random.Random(RANDOM_SEED)
    testable = rng.sample(pool, k=min(N_TEST, len(pool)))
    print(f"\nRandomly sampled {len(testable)} / {len(pool)} items  (seed={RANDOM_SEED})\n")

    results = []
    for i, item in enumerate(testable):
        frames_root = Path(item["_frames_root"])
        img_b64     = None
        for rel in item.get("frames", []):
            p = frames_root / rel
            if p.exists():
                img_b64 = encode_image_b64(p)
                break
        if img_b64 is None:
            from PIL import Image
            buf = BytesIO()
            Image.new("RGB", (224, 224), (100, 149, 237)).save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

        gt_raw = item.get("a", "")
        if isinstance(gt_raw, list):
            gt_raw = gt_raw[0] if gt_raw else ""
        query = item.get("q", "")

        try:
            resp = send_request(SERVER_HOST, SERVER_PORT, img_b64, query)
            raw  = resp.get("response", "")
            pred = parse_pred(raw)
            ok   = (pred == item["_action"])

            gt_value    = extract_value(gt_raw, item["_action"])
            pred_value  = extract_value(raw, pred) if pred else None
            # value_match is only meaningful when action is already correct
            value_match = (gt_value == pred_value) if (ok and gt_value is not None and pred_value is not None) else None

            results.append({
                "index":       i,
                "video_id":    item.get("video_id"),
                "dataset":     item["_dataset"],
                "question":    query,
                "gt_action":   item["_action"],
                "gt_raw":      gt_raw,
                "gt_value":    gt_value,
                "pred_action": pred,
                "pred_raw":    raw,
                "pred_value":  pred_value,
                "value_match": value_match,
                "correct":     ok,
            })
            mark = "✓" if ok else "✗"
            print(f"[{i+1:2d}/{len(testable)}] {mark}  gt={item['_action']:<12} "
                  f"pred={str(pred):<12} {item.get('video_id',''):<22}")
            print(f"         gt  raw: {gt_raw!r}")
            print(f"         pred raw: {raw!r}")
            if gt_value is not None:
                vmatch_str = "✓" if value_match else ("✗" if value_match is False else "?")
                print(f"         value: gt={gt_value}  pred={pred_value}  {vmatch_str}")
            print()
        except Exception as e:
            print(f"[{i+1:2d}/{len(testable)}] ERROR: {e}")
            break

    return results


def save_results(results):
    if not results:
        return
    n_ok     = sum(r["correct"] for r in results)
    accuracy = n_ok / len(results)
    print(f"Accuracy: {n_ok}/{len(results)}  ({100*accuracy:.1f}%)")

    value_items = [r for r in results if r.get("value_match") is not None]
    n_value_ok  = sum(r["value_match"] for r in value_items)
    if value_items:
        print(f"Value accuracy (action+value correct / all samples): "
              f"{n_value_ok}/{len(results)}  ({100*n_value_ok/len(results):.1f}%)")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "meta": {
            "timestamp":        ts,
            "server":           f"{SERVER_HOST}:{SERVER_PORT}",
            "n_sampled":        len(results),
            "random_seed":      RANDOM_SEED,
            "filter_dataset":   FILTER_DATASET,
            "filter_action":    FILTER_ACTION,
            "filter_keyword":   FILTER_KEYWORD,
            "accuracy":         round(accuracy, 4),
            "n_correct":        n_ok,
            "value_accuracy":   round(n_value_ok / len(results), 4) if value_items else None,
            "n_value_correct":  n_value_ok,
            "n_value_items":    len(value_items),
        },
        "results": results,
    }
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"test_{ts}_seed{RANDOM_SEED}_n{len(results)}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    all_data = load_datasets()
    filtered = filter_items(all_data)
    results  = batch_test(filtered)
    save_results(results)

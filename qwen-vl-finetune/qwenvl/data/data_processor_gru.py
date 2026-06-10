import json
import random
import logging
import re
import time
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_MOTION_TOKEN = "<motion>"

DEGREE_PATTERN = re.compile(r"(\d+)\s*degree", re.IGNORECASE)
CM_PATTERN = re.compile(r"(\d+)\s*cm", re.IGNORECASE)

STOP = 0
FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

local_rank = None


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def actions_to_motion_features(action_seq, theta0=0.0, step_m=0.25, turn_deg=15.0):
    """
    Convert integer action IDs into the same 7D motion features used in gru_train_final.ipynb.
    Feature order: [cum_x, cum_y, sin(theta), cos(theta), dyaw, is_forward, is_turn].
    """
    if not action_seq:
        return torch.zeros((1, 7), dtype=torch.float32)

    theta = theta0
    turn_rad = math.radians(turn_deg)
    cum_x, cum_y = 0.0, 0.0
    rows = []

    for a in action_seq:
        dx_local = 0.0
        dyaw = 0.0
        if a == FORWARD:
            dx_local = step_m
        elif a == TURN_LEFT:
            dyaw = turn_rad
        elif a == TURN_RIGHT:
            dyaw = -turn_rad

        theta += dyaw
        cum_x += dx_local * math.cos(theta)
        cum_y += dx_local * math.sin(theta)

        rows.append(
            [
                cum_x,
                cum_y,
                math.sin(theta),
                math.cos(theta),
                dyaw,
                1.0 if a == FORWARD else 0.0,
                1.0 if a in (TURN_LEFT, TURN_RIGHT) else 0.0,
            ]
        )

    return torch.tensor(rows, dtype=torch.float32)


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


class _MissingImage(PILImage.Image):
    """Sentinel subclass returned when an image/frame file is not found."""
    pass


def _sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []
    n = min(total_frames, num_frames)
    return np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()


def _extract_video_frames_at(video_path: str, indices: Sequence[int]) -> List[PILImage.Image]:
    """Load specific frame indices from an mp4. Black-frame fallback on miss."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[PILImage.Image] = []
    if total == 0:
        cap.release()
        logging.warning(f"Missing or unreadable video: {video_path}, using black frames.")
        black = PILImage.new("RGB", (224, 224))
        black.__class__ = _MissingImage
        return [black for _ in indices]
    for raw_idx in indices:
        idx = max(0, min(int(raw_idx), total - 1))
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


def _extract_video_frames(video_path: str, num_frames: int) -> List[PILImage.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        logging.warning(f"Missing or unreadable video: {video_path}, using black frames.")
        black = PILImage.new("RGB", (224, 224))
        black.__class__ = _MissingImage
        return [black] * num_frames
    indices = _sample_frame_indices(total, num_frames)
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
    while len(frames) < num_frames:
        frames.insert(0, PILImage.new("RGB", (224, 224)))
    return frames[:num_frames]


# Number of frames sampled for LLaVA-style (sharegpt4v / sharegptvideo) samples,
# matching the base data_processor (NUM_HISTORICAL_FRAMES + 1).
NUM_HISTORICAL_FRAMES = 7


def _strip_visual_tokens(text: str) -> str:
    """Remove visual placeholder tokens from text; images are attached separately."""
    for tok in ("<image>", "<video>", "<gru>", DEFAULT_MOTION_TOKEN):
        text = text.replace(tok, "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_conversations(conversations: List[Dict]) -> List[Dict]:
    out = []
    for turn in conversations:
        speaker = turn.get("from", "")
        if speaker in {"human", "user"}:
            out.append({"from": "human", "value": str(turn.get("value", ""))})
        elif speaker in {"gpt", "assistant"}:
            out.append({"from": "gpt", "value": str(turn.get("value", ""))})
    return out


def _load_frame_dir(frame_dir: Path, num_frames: int) -> Tuple[List[PILImage.Image], bool]:
    """Load uniformly sampled frames from a directory of pre-extracted image files."""
    exts = {".jpeg", ".jpg", ".png"}
    try:
        files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)
    except (FileNotFoundError, NotADirectoryError):
        files = []
    if not files:
        return [PILImage.new("RGB", (224, 224))] * num_frames, True
    indices = _sample_frame_indices(len(files), num_frames)
    frames = [PILImage.open(files[i]).convert("RGB") for i in indices]
    return frames, False


def split_video_id(video_id: str) -> Tuple[str, int]:
    if not isinstance(video_id, str) or "-" not in video_id:
        return str(video_id), 0
    traj, step = video_id.rsplit("-", 1)
    try:
        return traj, int(step)
    except ValueError:
        return traj, 0


def action_codes_from_answer(answer: str) -> List[int]:
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


def normalize_action_answer(answer: str) -> str:
    text = (answer or "").strip()
    low = text.lower()
    prefix = "the next action is "
    if low.startswith(prefix):
        text = text[len(prefix) :].strip()
    return text.rstrip(".")


def select_frame_slots(frames: Sequence[str], slots: int = 8) -> List[Tuple[int, str]]:
    if isinstance(frames, str):
        frames = [frames]
    frames = list(frames or [])
    if len(frames) == 0:
        raise ValueError("Sample has no frames")

    n_slots = max(1, int(slots))
    indices = np.linspace(0, len(frames) - 1, num=n_slots, dtype=int).tolist()
    return [(int(i), frames[int(i)]) for i in indices]


def _extract_frame_index(frame_rel: str) -> Optional[int]:
    match = re.search(r"frame_(\d+)\.(?:jpg|jpeg|png|webp)$", str(frame_rel), re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"frame_(\d+)$", str(frame_rel), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Build chat messages for a sample.

    Returns:
        (messages, has_gru): has_gru=True only for VLN samples (R2R/RxR/Human) that
        carry a real per-step action stream the trajectory GRU can encode. For all
        other datasets (EnvDrop / ScanQA / video_chatgpt / sharegpt*) we return
        has_gru=False and omit the <motion> token entirely so the model skips
        GRU injection for that row.
    """
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful navigation assistant."}],
    }

    # ---- Native R2R / RxR / Human schema: {video_id, q, a, frames} (frame folder) ----
    # This is the only branch that uses real per-step actions -> has_gru=True.
    if (
        "conversations" not in item
        and "q" in item
        and "a" in item
        and "frames" in item
    ):
        n_slots = int(item.get("_gru_history_slots", 8))
        selected = select_frame_slots(item.get("frames") or [], slots=n_slots)

        user_content = [
            {
                "type": "text",
                "text": (
                    "You are a robot agent programmed for navigation tasks. "
                    "Below are historical observations consisting of an image and "
                    "trajectory memory tokens."
                ),
            },
        ]

        for slot_idx, (_, frame_rel) in enumerate(selected):
            if slot_idx < max(0, n_slots - 1):
                title = f"History {slot_idx + 1}"
            else:
                title = "Current observation"
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"{title}\\n"
                        f"Frame id: {frame_rel}\\n"
                        f"Trajectory memory tokens: {DEFAULT_MOTION_TOKEN}"
                    ),
                }
            )
            user_content.append(
                {"type": "image", "image": _make_abs_paths(base_path, frame_rel)}
            )

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Instruction: {str(item.get('q', ''))}\\n\\n"
                    "Predict the next navigation action. Valid answers should be concise, "
                    "for example: 'turn left 15 degrees', 'turn right 30 degrees', "
                    "'move forward 75 cm', or 'stop'."
                ),
            }
        )

        return [
            system_message,
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": normalize_action_answer(str(item.get("a", "")))}
                ],
            },
        ], True

    # ---- EnvDrop motion schema: {video_id, q, frames, motion} + per-frame action codes ----
    # Same layout as R2R: 8 uniformly sampled frames, each preceded by a <motion>
    # token whose GRU feature is the action prefix up to that frame index. The
    # assistant target is the natural-language trajectory description in `q`.
    if (
        "conversations" not in item
        and "frames" in item
        and "motion" in item
        and "q" in item
        and "a" not in item
    ):
        n_slots = int(item.get("_gru_history_slots", 8))
        frames = item.get("frames") or []
        selected = select_frame_slots(frames, slots=n_slots)
        sampled_indices = [idx for idx, _ in selected]

        video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
        pil_frames = _extract_video_frames_at(video_path, sampled_indices)
        while len(pil_frames) < n_slots:
            pil_frames.append(PILImage.new("RGB", (224, 224)))

        user_content = [
            {
                "type": "text",
                "text": (
                    "Assume you are a robot designed for navigation. You are provided "
                    "with captured images sequences "
                ),
            },
        ]
        for pil_frame in pil_frames[:n_slots]:
            user_content.append({"type": "text", "text": DEFAULT_MOTION_TOKEN})
            user_content.append({"type": "image", "image": pil_frame})
        user_content.append(
            {
                "type": "text",
                "text": (
                    ". Based on this image sequence, please describe the navigation "
                    "trajectory of the robot."
                ),
            }
        )

        return [
            system_message,
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(item.get("q", ""))}],
            },
        ], True

    # ---- EnvDrop legacy schema: {video_id (int), instruction} + single .mp4 ----
    if (
        "conversations" not in item
        and "frames" not in item
        and "instruction" in item
        and "q" not in item
    ):
        video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
        pil_frames = _extract_video_frames(video_path, 8)
        content = [
            {
                "type": "text",
                "text": (
                    "Assume you are a robot designed for navigation. You are provided "
                    "with captured images sequences"
                ),
            },
            *[{"type": "image", "image": f} for f in pil_frames],
            {
                "type": "text",
                "text": (
                    ". Based on this image sequence, please describe the navigation "
                    "trajectory of the robot."
                ),
            },
        ]
        return [
            system_message,
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(item["instruction"])}],
            },
        ], False

    # ---- ScanQA / video_chatgpt schema: {video_id, q, a} + .mp4, no frames key ----
    if (
        "conversations" not in item
        and "frames" not in item
        and "q" in item
        and "a" in item
    ):
        video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
        pil_frames = _extract_video_frames(video_path, 8)
        answer = random.choice(item["a"]) if isinstance(item["a"], list) else item["a"]
        content = [
            *[{"type": "image", "image": f} for f in pil_frames],
            {"type": "text", "text": str(item["q"])},
        ]
        return [
            system_message,
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
        ], False

    # ---- LLaVA-style multi-turn conversations (sharegpt4v / sharegptvideo / etc.) ----
    # Mirror the base (non-GRU) data processor: load every visual as PIL image frames
    # and prepend them to the first user turn. ShareGPT4V stores image files;
    # ShareGPTVideo stores pre-extracted frames under frames/{video}/. We never emit a
    # {"type": "video"} item, because HF's load_video rejects a frame-dir / list path
    # (the cause of the "Incorrect format used for video" crash). has_gru=False.
    turns = _normalize_conversations(item["conversations"])
    images: List[PILImage.Image] = []

    if "image" in item:
        image_files = item["image"] if isinstance(item["image"], list) else [item["image"]]
        for img_file in image_files:
            img_path = base_path / str(img_file)
            try:
                images.append(PILImage.open(img_path).convert("RGB"))
            except (FileNotFoundError, OSError):
                black = PILImage.new("RGB", (224, 224))
                black.__class__ = _MissingImage
                images.append(black)
    elif "video" in item:
        # ShareGPTVideo — frames pre-extracted in frames/{video}/ directory.
        frame_dir = base_path / str(item["video"])
        images, _ = _load_frame_dir(frame_dir, NUM_HISTORICAL_FRAMES + 1)

    messages = [system_message]
    for i, turn in enumerate(turns):
        role = "user" if turn["from"] == "human" else "assistant"
        value = turn["value"]
        if role == "assistant":
            content = [{"type": "text", "text": value}]
        elif i == 0:
            # First user turn: strip visual/motion tokens and prepend all images.
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": _strip_visual_tokens(value)})
        else:
            content = [{"type": "text", "text": _strip_visual_tokens(value)}]
        messages.append({"role": role, "content": content})

    return messages, False


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages, has_gru = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    full_result["_has_gru"] = bool(has_gru)

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            ann_path = str(data["annotation_path"])
            ann_exists = Path(ann_path).exists()
            rank0_print(
                f"[GRU-Data] Loading annotations from {ann_path} (exists={ann_exists})"
            )
            if not ann_exists:
                raise FileNotFoundError(
                    f"Dataset annotation file not found: {ann_path}. "
                    f"dataset_use={data_args.dataset_use}"
                )

            file_format = ann_path.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(ann_path)
            else:
                annotations = json.load(open(ann_path, "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")


        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.gru_history_slots = max(1, int(getattr(data_args, "gru_history_slots", 8)))
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        self.traj_cumulative_actions = {}
        self._build_traj_action_index()

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    def _build_traj_action_index(self):
        traj_steps: Dict[str, Dict[int, List[int]]] = {}
        for ann in self.list_data_dict:
            if not isinstance(ann, dict):
                continue
            if "conversations" in ann:
                continue
            if "video_id" not in ann or "a" not in ann:
                continue

            traj, step = split_video_id(str(ann.get("video_id", "")))
            if traj not in traj_steps:
                traj_steps[traj] = {}
            traj_steps[traj][step] = action_codes_from_answer(str(ann.get("a", "")))

        for traj, step_map in traj_steps.items():
            running: List[int] = []
            cumulative: Dict[int, List[int]] = {}
            for step in sorted(step_map):
                running.extend(step_map[step])
                cumulative[step] = list(running)
            self.traj_cumulative_actions[traj] = cumulative

    def _cumulative_actions_until_inclusive(self, traj: str, step_inclusive: int) -> List[int]:
        if step_inclusive < 0:
            return []

        cumulative = self.traj_cumulative_actions.get(traj, {})
        if not cumulative:
            return []

        use_step = None
        for s in sorted(cumulative.keys()):
            if s <= step_inclusive:
                use_step = s
            else:
                break

        if use_step is None:
            return []
        return cumulative.get(use_step, [])

    def _runtime_native_gru_features(self, source_item: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
        slots = self.gru_history_slots
        frames = source_item.get("frames") or []
        selected = select_frame_slots(frames, slots=slots)
        frame_ids = [frame_rel for _, frame_rel in selected]

        traj, _ = split_video_id(str(source_item.get("video_id", "")))

        # Use the actual frame number for the cutoff rather than an interpolated slot anchor.
        # This keeps "History N -> frame_k" aligned with the real prefix available at frame k,
        # while still clamping to the observed trajectory length.
        observed_steps: List[int] = []
        for fallback_idx, frame_rel in selected:
            frame_idx = _extract_frame_index(frame_rel)
            if frame_idx is None:
                frame_idx = int(fallback_idx)
            observed_steps.append(max(0, int(frame_idx)))

        slot_prefixes: List[torch.Tensor] = []
        slot_lengths: List[int] = []
        for observed_step in observed_steps:
            # Strict no-future rule: slot at observed step t only sees actions before t.
            action_seq = self._cumulative_actions_until_inclusive(traj, observed_step - 1)
            # Frame-aligned cap: do not allow more GRU steps than frame transitions
            # represented by this history image cutoff.
            max_prefix_len = max(1, int(observed_step) - 1)
            if len(action_seq) > max_prefix_len:
                action_seq = action_seq[:max_prefix_len]
            if len(action_seq) == 0:
                prefix = actions_to_motion_features([STOP])
            else:
                prefix = actions_to_motion_features(action_seq)
            slot_prefixes.append(prefix)
            slot_lengths.append(int(prefix.size(0)))

        max_t = max(slot_lengths) if slot_lengths else 1
        padded_prefixes: List[torch.Tensor] = []
        for prefix in slot_prefixes:
            if prefix.size(0) < max_t:
                pad = torch.zeros((max_t - prefix.size(0), prefix.size(1)), dtype=prefix.dtype)
                prefix = torch.cat([prefix, pad], dim=0)
            padded_prefixes.append(prefix)

        return (
            torch.stack(padded_prefixes, dim=0),
            torch.tensor(slot_lengths, dtype=torch.long),
            frame_ids,
            observed_steps,
        )

    def _runtime_envdrop_gru_features(
        self, source_item: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
        """Build (slots, max_t, 7) GRU prefixes for an EnvDrop motion sample.

        Each sample carries the full per-frame action stream inline in `motion`,
        so prefixes are taken directly from `motion[:frame_idx]` with the strict
        no-future rule (slot at frame t only sees actions before t).
        """
        slots = self.gru_history_slots
        frames = source_item.get("frames") or []
        motion = source_item.get("motion") or []

        selected = select_frame_slots(frames, slots=slots)
        sampled_indices = [int(idx) for idx, _ in selected]
        frame_ids = [frame_rel for _, frame_rel in selected]

        slot_prefixes: List[torch.Tensor] = []
        slot_lengths: List[int] = []
        for frame_idx in sampled_indices:
            cutoff = max(0, min(frame_idx, len(motion)))
            action_seq = [int(a) for a in motion[:cutoff]]
            if not action_seq:
                prefix = actions_to_motion_features([STOP])
            else:
                prefix = actions_to_motion_features(action_seq)
            slot_prefixes.append(prefix)
            slot_lengths.append(int(prefix.size(0)))

        max_t = max(slot_lengths) if slot_lengths else 1
        padded: List[torch.Tensor] = []
        for prefix in slot_prefixes:
            if prefix.size(0) < max_t:
                pad = torch.zeros((max_t - prefix.size(0), prefix.size(1)), dtype=prefix.dtype)
                prefix = torch.cat([prefix, pad], dim=0)
            padded.append(prefix)

        return (
            torch.stack(padded, dim=0),
            torch.tensor(slot_lengths, dtype=torch.long),
            frame_ids,
            sampled_indices,
        )

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # missing file: skip forward immediately, no sleep
        next_index = i
        for _ in range(num_base_retries):
            try:
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]
                return self.item_fn(sources)
            except FileNotFoundError as e:
                print(f"[Skip] Missing file for sample {next_index}, trying next. {e}")
                next_index = min(next_index + 1, len(self.list_data_dict) - 1)

        # transient error: retry current sample with sleep
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                return self.item_fn(sources)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        raise RuntimeError(f"Failed to fetch any valid sample after retries, last index {i}.")

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        source_item = sources[0] if isinstance(sources, list) and len(sources) > 0 else {}
        is_native = (
            isinstance(source_item, dict)
            and "conversations" not in source_item
            and "q" in source_item
            and "a" in source_item
            and "frames" in source_item
        )
        is_envdrop_motion = (
            isinstance(source_item, dict)
            and "conversations" not in source_item
            and "frames" in source_item
            and "motion" in source_item
            and "q" in source_item
            and "a" not in source_item
        )

        preprocess_sources = sources
        if is_native or is_envdrop_motion:
            patched = dict(source_item)
            patched["_gru_history_slots"] = self.gru_history_slots
            preprocess_sources = [patched]

        data_dict = preprocess_qwen_visual(
            preprocess_sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        # Trust the has_gru flag from _build_messages: it is the single source of
        # truth for whether this sample carries a real per-step action stream.
        has_gru = bool(data_dict.pop("_has_gru", is_native or is_envdrop_motion))

        if has_gru and is_envdrop_motion:
            gru_features, gru_lengths, frame_ids, step_targets = self._runtime_envdrop_gru_features(source_item)
            data_dict["frame_ids"] = frame_ids
            data_dict["frame_step_targets"] = step_targets
        elif has_gru:
            gru_features, gru_lengths, frame_ids, step_targets = self._runtime_native_gru_features(source_item)
            data_dict["frame_ids"] = frame_ids
            data_dict["frame_step_targets"] = step_targets
        elif "gru" in source_item:
            # Legacy LLaVA-style conversations with an explicit "gru" action list.
            raw_actions = source_item.get("gru", [])
            if not isinstance(raw_actions, list):
                raw_actions = []
            raw_actions = [int(a) for a in raw_actions if isinstance(a, (int, float))]

            min_seq_len = max(1, int(getattr(self.data_args, "gru_min_seq_len", 1)))
            if len(raw_actions) < min_seq_len:
                if getattr(self.data_args, "gru_fallback_to_stop", True):
                    raw_actions = [STOP] * min_seq_len
                else:
                    raise ValueError(
                        f"Sample has short GRU sequence len={len(raw_actions)} < {min_seq_len}"
                    )

            gru_features = actions_to_motion_features(raw_actions).unsqueeze(0)
            gru_lengths = torch.tensor([int(gru_features.size(1))], dtype=torch.long)
            has_gru = True
        else:
            # Non-VLN sample (EnvDrop/ScanQA/sharegpt*). No <motion> token was emitted
            # by _build_messages, so the model will see zero motion positions and skip
            # injection. We still emit a dummy 1x1x7 feature so the collator can stack.
            gru_features = torch.zeros((1, 1, 7), dtype=torch.float32)
            gru_lengths = torch.zeros((1,), dtype=torch.long)

        data_dict["gru_features"] = gru_features
        data_dict["gru_lengths"] = gru_lengths
        data_dict["gru_length"] = torch.tensor(int(gru_features.size(0)), dtype=torch.long)
        data_dict["has_gru"] = torch.tensor(1 if has_gru else 0, dtype=torch.long)

        # Motion-token diagnostics used by debug scripts and forward alignment checks.
        motion_token_text = getattr(self.data_args, "motion_token_text", DEFAULT_MOTION_TOKEN)
        motion_token_id = self.processor.tokenizer.convert_tokens_to_ids(motion_token_text)
        if motion_token_id is None:
            motion_token_id = -1
        token_row = data_dict["input_ids"][0]
        motion_positions = (token_row == motion_token_id).nonzero(as_tuple=False).squeeze(-1)
        data_dict["motion_token_id"] = torch.tensor(int(motion_token_id), dtype=torch.long)
        data_dict["motion_positions"] = motion_positions.to(dtype=torch.long)
        data_dict["motion_token_count"] = torch.tensor(int(motion_positions.numel()), dtype=torch.long)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


def collate_gru_prefixes(instances: Sequence[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    gru_features = [instance["gru_features"] for instance in instances]
    gru_lengths = [instance.get("gru_lengths") for instance in instances]

    max_slots = max(int(feat.size(0)) for feat in gru_features)
    max_t = max(int(feat.size(1)) for feat in gru_features)
    feat_dim = int(gru_features[0].size(2))

    feat_batch = torch.zeros(
        (len(instances), max_slots, max_t, feat_dim),
        dtype=gru_features[0].dtype,
    )
    len_batch = torch.zeros((len(instances), max_slots), dtype=torch.long)

    for i, feat in enumerate(gru_features):
        s, t, _ = feat.shape
        feat_batch[i, :s, :t, :] = feat
        gl = gru_lengths[i]
        if gl is None:
            gl = torch.full((s,), int(t), dtype=torch.long)
        len_batch[i, :s] = gl[:s]

    return feat_batch, len_batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        if all("gru_features" in instance for instance in instances):
            feat_batch, len_batch = collate_gru_prefixes(instances)
            batch["gru_features"] = feat_batch
            batch["gru_lengths"] = len_batch

        if all("has_gru" in instance for instance in instances):
            batch["has_gru"] = torch.stack(
                [instance["has_gru"].to(dtype=torch.long) for instance in instances]
            )

        if all("motion_token_id" in instance for instance in instances):
            batch["motion_token_id"] = int(instances[0]["motion_token_id"])
        if all("motion_token_count" in instance for instance in instances):
            batch["motion_token_count"] = torch.tensor(
                [int(instance["motion_token_count"]) for instance in instances], dtype=torch.long
            )
        if all("motion_positions" in instance for instance in instances):
            batch["motion_positions"] = [instance["motion_positions"].tolist() for instance in instances]

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        if all("gru_features" in instance for instance in instances):
            feat_batch, len_batch = collate_gru_prefixes(instances)
            batch["gru_features"] = feat_batch
            batch["gru_lengths"] = len_batch

        if all("has_gru" in instance for instance in instances):
            batch["has_gru"] = torch.stack(
                [instance["has_gru"].to(dtype=torch.long) for instance in instances]
            )

        if all("motion_token_id" in instance for instance in instances):
            batch["motion_token_id"] = int(instances[0]["motion_token_id"])
        if all("motion_token_count" in instance for instance in instances):
            batch["motion_token_count"] = torch.tensor(
                [int(instance["motion_token_count"]) for instance in instances], dtype=torch.long
            )
        if all("motion_positions" in instance for instance in instances):
            batch["motion_positions"] = [instance["motion_positions"].tolist() for instance in instances]

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass

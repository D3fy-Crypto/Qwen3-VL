import json
import random
import logging
import re
import time
import itertools
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

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


class _MissingImage(PILImage.Image):
    """Sentinel subclass returned when an image file is not found."""
    pass


def _make_abs_paths(base: Path, files: str) -> PILImage.Image:
    try:
        return PILImage.open((base / files).resolve()).convert("RGB")
    except FileNotFoundError:
        logging.warning(f"Missing image: {base / files}, using black frame.")
        img = PILImage.new("RGB", (224, 224))
        img.__class__ = _MissingImage
        return img


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


NUM_HISTORICAL_FRAMES = 7


def _strip_visual_tokens(text: str) -> str:
    text = text.replace("<image>", "").replace("<video>", "").replace("<motion>", "")
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
    """Load uniformly sampled frames from a directory of image files."""
    exts = {".jpeg", ".jpg", ".png"}
    files = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        return [PILImage.new("RGB", (224, 224))] * num_frames, True
    indices = _sample_frame_indices(len(files), num_frames)
    frames = [PILImage.open(files[i]).convert("RGB") for i in indices]
    return frames, False


def _sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Uniformly sample num_frames indices from total_frames.
    Caller is responsible for padding if total_frames < num_frames.
    """
    if total_frames <= 0 or num_frames <= 0:
        return []
    n = min(total_frames, num_frames)
    return np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()


def _extract_video_frames(video_path: str, num_frames: int) -> List[PILImage.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        # File not found or unreadable — return sentinel frames so callers can detect it.
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
    return frames


def _build_messages(item: Dict[str, Any], base_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    system_prompt = "You are a helpful navigation assistant."
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    # ShareGPT4V / ShareGPTVideo: LLaVA-style multi-turn conversations format
    if "conversations" in item:
        turns = _normalize_conversations(item["conversations"])
        images: List[PILImage.Image] = []
        has_missing = False

        if "image" in item:
            # ShareGPT4V — single image or list of image paths
            image_files = item["image"] if isinstance(item["image"], list) else [item["image"]]
            for img_file in image_files:
                img_path = base_path / img_file
                try:
                    images.append(PILImage.open(img_path).convert("RGB"))
                except FileNotFoundError:
                    logging.warning(f"Missing image: {img_path}, using black frame.")
                    black = PILImage.new("RGB", (224, 224))
                    black.__class__ = _MissingImage
                    images.append(black)
                    has_missing = True
        elif "video" in item:
            # ShareGPTVideo — frames pre-extracted in frames/{video}/ directory
            frame_dir = base_path / item["video"]
            images, has_missing = _load_frame_dir(frame_dir, NUM_HISTORICAL_FRAMES + 1)

        for i, turn in enumerate(turns):
            role = "user" if turn["from"] == "human" else "assistant"
            value = turn["value"]
            if role == "assistant":
                content = [{"type": "text", "text": value}]
            elif i == 0:
                # First user turn: strip visual tokens and prepend all images
                clean = _strip_visual_tokens(value)
                content = [{"type": "image", "image": img} for img in images]
                content.append({"type": "text", "text": clean})
            else:
                content = [{"type": "text", "text": _strip_visual_tokens(value)}]
            messages.append({"role": role, "content": content})

        return messages, has_missing

    if "frames" not in item:
        # EnvDrop format: {video_id (int), instruction}
        # video filename = video_id directly  (e.g. video_id=1 -> 1.mp4)
        if "instruction" in item and "q" not in item:
            video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
            pil_frames = _extract_video_frames(video_path, NUM_HISTORICAL_FRAMES + 1)
            has_missing = any(isinstance(f, _MissingImage) for f in pil_frames)
            content = [
                {"type": "text", "text": "Assume you are a robot designed for navigation. "
                    "You are provided with captured images sequences"},
                *[{"type": "image", "image": f} for f in pil_frames],
                {"type": "text", "text": ". Based on this image sequence, please describe "
                    "the navigation trajectory of the robot."},
            ]
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["instruction"]}]})
            return messages, has_missing

        # ScanQA / Video-ChatGPT format: video_id + q + a (list), no frames key
        video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
        pil_frames = _extract_video_frames(video_path, NUM_HISTORICAL_FRAMES + 1)
        has_missing = any(isinstance(f, _MissingImage) for f in pil_frames)
        answer = random.choice(item["a"]) if isinstance(item["a"], list) else item["a"]
        content = [
            *[{"type": "image", "image": f} for f in pil_frames],
            {"type": "text", "text": item["q"]},
        ]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages, has_missing

    frames = item["frames"]
    if len(frames) < 1:
        raise ValueError(f"item has no frames: {item.get('video_id', '')}")
    current = frames[-1]
    historical_pool = frames[:-1] if len(frames) > 1 else []

    # Uniformly sample up to NUM_HISTORICAL_FRAMES from pool; prepend black frames if not enough.
    indices = _sample_frame_indices(len(historical_pool), NUM_HISTORICAL_FRAMES)
    loaded_historical = [_make_abs_paths(base_path, historical_pool[i]) for i in indices]
    pad_count = NUM_HISTORICAL_FRAMES - len(loaded_historical)
    black_frame = PILImage.new("RGB", (224, 224))
    loaded_historical = [black_frame] * pad_count + loaded_historical

    loaded_current = _make_abs_paths(base_path, current)
    has_missing = any(isinstance(f, _MissingImage) for f in loaded_historical) or isinstance(loaded_current, _MissingImage)

    content = [
        {"type": "text", "text": "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations:"},
        *[{"type": "image", "image": f} for f in loaded_historical],
        {"type": "text", "text": "and current observation:"},
        {"type": "image", "image": loaded_current},
        {"type": "text", "text": (
            f"Your assigned task is: {item['q']}\n"
            "Analyze this series of images to decide your next move, which could involve "
            "turning left or right by a specific degree, moving forward a certain distance, "
            "or stop if the task is completed."
        )},
    ]
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": item["a"]}]})

    return messages, has_missing


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))

    # Build the multi-turn message list and load PIL images.
    # has_missing=True means at least one image file was not found on disk.
    messages, has_missing = _build_messages(source, base_path)

    # Tokenize the full conversation (system + user + assistant turns).
    # Returns input_ids, attention_mask, pixel_values, image_grid_thw, etc.
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Start with all positions masked — only assistant tokens will be unmasked below.
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    # Unmask every assistant response segment so only those tokens contribute to loss.
    # Token 77091 = "assistant" (inside <|im_start|>assistant\n)
    # Token 151645 = <|im_end|>
    # Turn layout: <|im_start|> assistant \n <response...> <|im_end|> \n
    #                            ^77091    ^+2=ans_start              ^ans_end
    # We unmask [ans_start : ans_end+2] to include the response and closing <|im_end|>\n.
    ids = input_ids[0]
    asst_positions = (ids == 77091).nonzero(as_tuple=True)[0]   # one per assistant turn
    end_positions  = (ids == 151645).nonzero(as_tuple=True)[0]  # one per im_end token

    for asst_pos in asst_positions:
        ans_start = asst_pos + 2
        # find the first <|im_end|> that comes after this assistant token
        later_ends = end_positions[end_positions > asst_pos]
        if len(later_ends) == 0:
            continue
        ans_end = later_ends[0].item()
        labels[0, ans_start : ans_end + 2] = ids[ans_start : ans_end + 2] # +2 includes the <|im_end|> and following \n

    # If any image was missing, suppress the entire sample from contributing to
    # the loss — corrupt visual context should not reinforce wrong predictions.
    if has_missing:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

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
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
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
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

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
        data_dict = preprocess_qwen_visual(
            sources,
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

        # ====== Debugging code to verify tokenization and alignment ======
        # text = self.processor.tokenizer.decode(
        #     data_dict["input_ids"][0], skip_special_tokens=False
        # )

        # labels = data_dict["labels"][0]
        # labels = [
        #     tid if tid != -100 else self.processor.tokenizer.pad_token_id
        #     for tid in labels
        # ]
        # label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
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
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers

from . import data_list
from .data_processor import (
    IGNORE_INDEX,
    NUM_HISTORICAL_FRAMES,
    _extract_video_frames,
    _make_abs_paths,
    _sample_frame_indices,
    pad_and_cat,
    update_processor_pixels,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# GRU encoding: raw {0,1,2,3} integer sequence → one-hot tensor
# ---------------------------------------------------------------------------

def encode_gru_sequence(gru_field: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of action integers from {0,1,2,3} into a one-hot float tensor.

    Returns:
        gru_features: [T, 4] float32 one-hot tensor  (T=1 if gru_field is empty)
        gru_lengths:  scalar tensor with the true sequence length
    """
    if not gru_field:
        return torch.zeros(1, 4, dtype=torch.float32), torch.tensor([1])

    ids = torch.clamp(torch.tensor(gru_field, dtype=torch.long), 0, 3)
    one_hot = F.one_hot(ids, num_classes=4).float()  # [T, 4]
    return one_hot, torch.tensor([len(gru_field)])


# ---------------------------------------------------------------------------
# Message builder — long-horizon system prompt, same image structure as SFT
# ---------------------------------------------------------------------------

def _build_messages_gru_sft(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are an advanced navigation robot with long-horizon planning capability. "
        "Your trajectory is encoded as a sequence and provided to you as spatial memory. "
        "Use this memory alongside your visual observations to reason across multiple "
        "steps and complete navigation tasks."
    )
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    # ScanQA / video format (no "frames" key)
    if "frames" not in item:
        video_path = str((base_path / f"{item['video_id']}.mp4").resolve())
        pil_frames = _extract_video_frames(video_path, NUM_HISTORICAL_FRAMES + 1)
        answer = random.choice(item["a"]) if isinstance(item["a"], list) else item["a"]
        content = [
            *[{"type": "image", "image": f} for f in pil_frames],
            {"type": "text", "text": item["q"]},
        ]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    # Standard navigation format: frames / q / a
    frames = item["frames"]
    if len(frames) < 1:
        raise ValueError(f"item has no frames: {item.get('video_id', '')}")

    current = frames[-1]
    historical_pool = frames[:-1] if len(frames) > 1 else [frames[0]]
    indices = _sample_frame_indices(len(historical_pool), NUM_HISTORICAL_FRAMES)
    historical = [historical_pool[i] for i in indices]

    content = [
        {"type": "text", "text": "Your trajectory history is encoded in your context. Historical observations:"},
        *[{"type": "image", "image": _make_abs_paths(base_path, f)} for f in historical],
        {"type": "text", "text": "Current observation:"},
        {"type": "image", "image": _make_abs_paths(base_path, current)},
        {"type": "text", "text": (
            f"Your assigned task is: {item['q']}\n"
            "Using your navigation history and current observation, plan your next action "
            "to make progress toward the long-horizon goal."
        )},
    ]
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": item["a"]}]})
    return messages


# ---------------------------------------------------------------------------
# Preprocessing — same as preprocess_qwen_visual but uses new message builder
# ---------------------------------------------------------------------------

def preprocess_qwen_visual_gru_sft(sources, processor) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages_gru_sft(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:  # assistant start token
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:  # end token
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


# ---------------------------------------------------------------------------
# Dataset — extends LazySupervisedDataset, adds GRU feature extraction
# ---------------------------------------------------------------------------

class LazySupervisedDatasetGRUSFT(LazySupervisedDataset):
    """Supervised fine-tuning dataset with GRU action-sequence modality."""

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # Use GRU-SFT message builder (long-horizon system prompt)
        data_dict = preprocess_qwen_visual_gru_sft(sources, self.processor)

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
            video_grid_thw=(torch.cat(video_grid_thw, dim=0) if video_grid_thw else None),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        # Decode labels for debug (same as parent)
        labels = data_dict["labels"][0]
        labels_decoded = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        self.processor.tokenizer.decode(labels_decoded, skip_special_tokens=False)

        # Extract and encode GRU action sequence
        source = sources[0] if isinstance(sources, list) else sources
        gru_field = source.get("gru", [])
        gru_features, gru_lengths = encode_gru_sequence(gru_field)
        data_dict["gru_features"] = gru_features   # [T, 4]
        data_dict["gru_lengths"] = gru_lengths      # scalar tensor

        return data_dict


# ---------------------------------------------------------------------------
# Collator — extends standard collator, pads and batches GRU tensors
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForGRUSFT(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning with GRU modality."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)

        gru_features = [inst["gru_features"] for inst in instances]   # list of [T, 4]
        gru_lengths = [inst["gru_lengths"] for inst in instances]      # list of [1]

        batch["gru_features"] = torch.nn.utils.rnn.pad_sequence(
            gru_features, batch_first=True
        )  # [B, T_max, 4]
        batch["gru_lengths"] = torch.cat(gru_lengths)  # [B]

        return batch


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_supervised_data_module_gru_sft(processor, data_args) -> Dict:
    """Make dataset and collator for GRU-SFT fine-tuning."""
    train_dataset = LazySupervisedDatasetGRUSFT(processor, data_args=data_args)
    data_collator = DataCollatorForGRUSFT(processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwenvl.train.argument import DataArguments
from qwenvl.data.data_processor_gru import make_supervised_data_module as make_gru_data_module
from qwenvl.data.data_processor_gru_sft import (
    make_supervised_data_module_gru_sft as make_gru_sft_data_module,
)


def infer_model_type(model_name_or_path: str) -> str:
    lower = model_name_or_path.lower()
    if "qwen3" in lower:
        return "qwen3vl"
    if "qwen2.5" in lower or "qwen2_5" in lower:
        return "qwen2.5vl"
    return "qwen2vl"


def add_motion_token(tokenizer, motion_token_text: str):
    added = 0
    if tokenizer.convert_tokens_to_ids(motion_token_text) == tokenizer.unk_token_id:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [motion_token_text]})
    motion_token_id = tokenizer.convert_tokens_to_ids(motion_token_text)
    return added, int(motion_token_id)


def decode_with_motion(tokenizer, ids, motion_token_id: int):
    ids = [int(x) for x in ids]
    out = []
    chunk = []

    def flush():
        nonlocal chunk
        if chunk:
            out.append(tokenizer.decode(chunk, skip_special_tokens=False))
            chunk = []

    for tok in ids:
        if tok == motion_token_id:
            flush()
            out.append("<motion>")
        else:
            chunk.append(tok)
    flush()
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Debug one GRU batch and print exact token placement inputs.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_use", required=True)
    parser.add_argument("--pipeline", choices=["gru", "gru_sft"], default="gru")
    parser.add_argument("--motion_token_text", default="<motion>")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dump_json", default="")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right",
    )

    added, motion_token_id = add_motion_token(tokenizer, args.motion_token_text)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    data_args = DataArguments(
        dataset_use=args.dataset_use,
        model_type=infer_model_type(args.model_name_or_path),
        motion_token_text=args.motion_token_text,
    )

    if args.pipeline == "gru":
        module = make_gru_data_module(processor, data_args)
    else:
        module = make_gru_sft_data_module(processor, data_args)

    dl = DataLoader(
        module["train_dataset"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=module["data_collator"],
    )
    batch = next(iter(dl))

    print("=== GRU Dataset Batch Debug ===")
    print(f"pipeline: {args.pipeline}")
    print(f"dataset_use: {args.dataset_use}")
    print(f"added_motion_tokens: {added}")
    print(f"motion_token_text: {args.motion_token_text}")
    print(f"motion_token_id: {motion_token_id}")

    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v).__name__}")

    ids0 = batch["input_ids"][0].tolist()
    labels0 = batch["labels"][0].tolist() if "labels" in batch else []
    attn0 = batch["attention_mask"][0].tolist() if "attention_mask" in batch else []
    motion_positions = [i for i, t in enumerate(ids0) if int(t) == motion_token_id]

    print("\n--- First sample exact inputs ---")
    print(f"input_ids_len: {len(ids0)}")
    print(f"motion_positions: {motion_positions}")
    print(f"motion_count: {len(motion_positions)}")
    print(f"gru_lengths[0]: {int(batch['gru_lengths'][0]) if 'gru_lengths' in batch else 'N/A'}")
    print(f"gru_features[0]_shape: {tuple(batch['gru_features'][0].shape) if 'gru_features' in batch else 'N/A'}")
    if "gru_features" in batch:
        print(f"gru_features[0]_head:\n{batch['gru_features'][0][:8]}")

    decoded = decode_with_motion(tokenizer, ids0, motion_token_id)
    print("\nDecoded input_ids[0] with <motion> placeholders:")
    print(decoded)

    if args.dump_json:
        payload = {
            "pipeline": args.pipeline,
            "dataset_use": args.dataset_use,
            "motion_token_text": args.motion_token_text,
            "motion_token_id": motion_token_id,
            "input_ids_0": ids0,
            "labels_0": labels0,
            "attention_mask_0": attn0,
            "motion_positions_0": motion_positions,
            "gru_lengths": batch.get("gru_lengths", torch.tensor([])).tolist()
            if torch.is_tensor(batch.get("gru_lengths", None))
            else [],
            "gru_features_0": batch["gru_features"][0].tolist() if "gru_features" in batch else [],
        }
        out = Path(args.dump_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote debug payload: {out}")


if __name__ == "__main__":
    main()

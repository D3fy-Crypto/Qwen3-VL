#!/usr/bin/env python3
import argparse
import html
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwenvl.train.argument import DataArguments
from qwenvl.models.gru_qwen import GRUQwenModel
from qwenvl.models.gru_sft_module import GRUSFTQwenModel
from qwenvl.data.data_processor_gru import (
    _build_messages,
    make_supervised_data_module as make_gru_data_module,
)
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


def build_batch(args, processor, tokenizer):
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
    return next(iter(dl))


def build_raw_prompt(args, processor):
    data_args = DataArguments(
        dataset_use=args.dataset_use,
        model_type=infer_model_type(args.model_name_or_path),
        motion_token_text=args.motion_token_text,
    )
    if args.pipeline == "gru":
        module = make_gru_data_module(processor, data_args)
    else:
        module = make_gru_sft_data_module(processor, data_args)

    dataset = module["train_dataset"]
    source = dataset.list_data_dict[0]
    if isinstance(source, list):
        source = source[0]

    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)
    raw_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {
        "source": source,
        "messages": messages,
        "raw_prompt": raw_prompt,
    }


def strip_vision_batch(batch, tokenizer):
    batch["pixel_values"] = None
    batch["image_grid_thw"] = None
    batch["pixel_values_videos"] = None
    batch["video_grid_thw"] = None

    image_token_id = 151655
    video_token_id = 151656
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for tok in (image_token_id, video_token_id):
        batch["input_ids"][batch["input_ids"] == tok] = int(pad_id)

    if "attention_mask" in batch and batch["attention_mask"] is not None:
        batch["attention_mask"] = batch["input_ids"].ne(int(pad_id)).long()

    return batch


def render_debug_html(payload):
        prompt = html.escape(str(payload.get("raw_prompt_0", "")))
        source_json = html.escape(json.dumps(payload.get("raw_source_0", {}), indent=2, ensure_ascii=False))
        messages_json = html.escape(json.dumps(payload.get("raw_messages_0", []), indent=2, ensure_ascii=False))
        report_json = html.escape(json.dumps(payload, indent=2, ensure_ascii=False))
        motion_positions = html.escape(json.dumps(payload.get("motion_positions_0", []), ensure_ascii=False))
        input_ids = html.escape(json.dumps(payload.get("input_ids_0", []), ensure_ascii=False))
        gru_features = html.escape(json.dumps(payload.get("gru_features_0", []), indent=2, ensure_ascii=False))
        logits_shape = html.escape(json.dumps(payload.get("logits_shape", None), ensure_ascii=False))
        loss = html.escape(str(payload.get("loss", None)))
        disable_vision = html.escape(str(payload.get("disable_vision", None)))

        return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>GRU Forward Debug Report</title>
    <style>
        :root {{
            --bg: #0b1020;
            --panel: #121a30;
            --panel-2: #0f172a;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #60a5fa;
            --accent-2: #34d399;
            --border: rgba(148, 163, 184, 0.22);
        }}
        body {{
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, rgba(96, 165, 250, 0.16), transparent 30%), var(--bg);
            color: var(--text);
        }}
        .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        .hero {{
            display: grid;
            gap: 12px;
            margin-bottom: 20px;
            padding: 20px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(18,26,48,0.96), rgba(15,23,42,0.96));
            box-shadow: 0 18px 60px rgba(0,0,0,0.28);
        }}
        h1 {{ margin: 0; font-size: 28px; letter-spacing: -0.02em; }}
        .meta {{ display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.22);
            color: var(--text);
            font-size: 13px;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
        .card {{
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.92);
            overflow: hidden;
        }}
        .card h2 {{ margin: 0; padding: 14px 16px; font-size: 16px; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02); }}
        .card .body {{ padding: 16px; }}
        pre {{
            margin: 0;
            padding: 14px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: #dbeafe;
            line-height: 1.4;
            font-size: 13px;
        }}
        .wide {{ grid-column: 1 / -1; }}
        .warn {{ color: #fca5a5; }}
        .ok {{ color: #86efac; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
        .stat {{ padding: 12px 14px; border: 1px solid var(--border); border-radius: 14px; background: rgba(255,255,255,0.02); }}
        .stat .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
        .stat .value {{ margin-top: 6px; font-size: 15px; word-break: break-word; }}
        @media (max-width: 1000px) {{ .grid, .stats {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <section class=\"hero\">
            <h1>GRU Forward Debug Report</h1>
            <div class=\"meta\">
                <span class=\"chip\">pipeline: {html.escape(str(payload.get("pipeline", "")))}</span>
                <span class=\"chip\">motion token: {html.escape(str(payload.get("motion_token_text", "")))}</span>
                <span class=\"chip\">motion id: {html.escape(str(payload.get("motion_token_id", "")))}</span>
                <span class=\"chip\">vision disabled: {disable_vision}</span>
                <span class=\"chip\">loss: {html.escape(str(loss))}</span>
            </div>
            <div class=\"stats\">
                <div class=\"stat\"><div class=\"label\">Logits Shape</div><div class=\"value\">{logits_shape}</div></div>
                <div class=\"stat\"><div class=\"label\">Motion Positions</div><div class=\"value\">{motion_positions}</div></div>
                <div class=\"stat\"><div class=\"label\">Input Length</div><div class=\"value\">{html.escape(str(len(payload.get("input_ids_0", []))))}</div></div>
                <div class=\"stat\"><div class=\"label\">GRU Length</div><div class=\"value\">{html.escape(str(payload.get("gru_lengths", [])))}</div></div>
            </div>
        </section>

        <div class=\"grid\">
            <section class=\"card wide\">
                <h2>Raw Prompt Before Tokenizer</h2>
                <div class=\"body\"><pre>{prompt}</pre></div>
            </section>

            <section class=\"card\">
                <h2>Raw Source Sample</h2>
                <div class=\"body\"><pre>{source_json}</pre></div>
            </section>

            <section class=\"card\">
                <h2>Chat Messages</h2>
                <div class=\"body\"><pre>{messages_json}</pre></div>
            </section>

            <section class=\"card wide\">
                <h2>Token IDs</h2>
                <div class=\"body\"><pre>{input_ids}</pre></div>
            </section>

            <section class=\"card\">
                <h2>GRU Features</h2>
                <div class=\"body\"><pre>{gru_features}</pre></div>
            </section>

            <section class=\"card\">
                <h2>Full JSON Payload</h2>
                <div class=\"body\"><pre>{report_json}</pre></div>
            </section>
        </div>
    </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Run one exact GRU forward pass and print token-placement internals.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_use", required=True)
    parser.add_argument("--pipeline", choices=["gru", "gru_sft"], default="gru")
    parser.add_argument("--motion_token_text", default="<motion>")
    parser.add_argument("--gru_checkpoint_path", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--dump_json", default="")
    parser.add_argument(
        "--disable_vision",
        action="store_true",
        default=False,
        help="Bypass vision branch for environments missing CUDA vision kernels.",
    )
    parser.add_argument(
        "--enable_vision",
        action="store_false",
        dest="disable_vision",
        help="Run with pixel_values/image tokens enabled.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

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

    if args.pipeline == "gru":
        model = GRUQwenModel(
            qwen_model_id=args.model_name_or_path,
            gru_checkpoint_path=args.gru_checkpoint_path or None,
            projector_k=1,
            motion_token_id=motion_token_id,
            device=args.device,
            dtype=torch_dtype,
            tune_qwen_vision=False,
            tune_qwen_lm=False,
            tune_projector=True,
        )
    else:
        model = GRUSFTQwenModel(
            qwen_model_id=args.model_name_or_path,
            projector_k=1,
            motion_token_id=motion_token_id,
            device=args.device,
            dtype=torch_dtype,
            tune_qwen_vision=False,
            tune_qwen_lm=False,
            tune_projector=True,
        )

    if added > 0:
        model.qwen.resize_token_embeddings(len(tokenizer))

    raw_debug = build_raw_prompt(args, processor)
    batch = build_batch(args, processor, tokenizer)
    vision_disabled = bool(args.disable_vision)

    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
                gru_features=batch.get("gru_features"),
                gru_lengths=batch.get("gru_lengths"),
            )
    except NotImplementedError as exc:
        if vision_disabled:
            raise
        message = str(exc)
        if "slow_conv3d_forward" not in message and "CUDA" not in message:
            raise
        print("[debug] Vision forward failed on this build; retrying with vision disabled.")
        batch = strip_vision_batch(batch, tokenizer)
        vision_disabled = True
        with torch.no_grad():
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
                gru_features=batch.get("gru_features"),
                gru_lengths=batch.get("gru_lengths"),
            )

    print("=== GRU Forward Debug ===")
    print(f"pipeline: {args.pipeline}")
    print(f"device: {args.device} dtype: {args.dtype}")
    print(f"motion_token_text: {args.motion_token_text} motion_token_id: {motion_token_id}")
    print(f"disable_vision: {vision_disabled}")
    print(f"logits_shape: {tuple(outputs['logits'].shape) if outputs.get('logits') is not None else None}")
    print(f"loss: {float(outputs['loss']) if outputs.get('loss') is not None else None}")

    ids0 = batch["input_ids"][0].tolist()
    motion_positions = [i for i, t in enumerate(ids0) if int(t) == motion_token_id]
    print(f"input_ids_len: {len(ids0)}")
    print(f"motion_positions: {motion_positions}")
    print(f"gru_lengths[0]: {int(batch['gru_lengths'][0])}")
    print(f"gru_features[0]_shape: {tuple(batch['gru_features'][0].shape)}")
    print("raw_prompt_0:")
    print(raw_debug["raw_prompt"])

    if args.dump_json:
        payload = {
            "pipeline": args.pipeline,
            "motion_token_text": args.motion_token_text,
            "motion_token_id": motion_token_id,
            "disable_vision": vision_disabled,
            "raw_source_0": raw_debug["source"],
            "raw_messages_0": raw_debug["messages"],
            "raw_prompt_0": raw_debug["raw_prompt"],
            "input_ids_0": ids0,
            "motion_positions_0": motion_positions,
            "gru_lengths": batch["gru_lengths"].tolist(),
            "gru_features_0": batch["gru_features"][0].tolist(),
            "logits_shape": list(outputs["logits"].shape) if outputs.get("logits") is not None else None,
            "loss": float(outputs["loss"]) if outputs.get("loss") is not None else None,
        }
        out = Path(args.dump_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        html_out = out.with_suffix(".html")
        html_out.write_text(render_debug_html(payload), encoding="utf-8")
        print(f"Wrote forward-debug payload: {out}")
        print(f"Wrote HTML report: {html_out}")


if __name__ == "__main__":
    main()

"""
FastAPI inference server for the original Qwen3-VL Instruct model.

Usage:
    python inference/server_base.py [--model-path PATH] [--host HOST] [--port PORT]

Default model: /home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct
Default port:  8000
"""

import argparse
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

model = None
processor = None
model_path = None


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL base model inference server")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def decode_image(image_value: str | Any) -> Any:
    """Accept a file path, base64 data URI, or PIL Image; return PIL Image or path string."""
    if isinstance(image_value, str) and image_value.startswith("data:image"):
        header, data = image_value.split(",", 1)
        img_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image_value  # file path — processor handles it directly


def prepare_messages(raw_messages: list[dict]) -> list[dict]:
    """Resolve image fields in messages (base64 → PIL.Image, paths unchanged)."""
    prepared = []
    for msg in raw_messages:
        content = []
        for block in msg.get("content", []):
            if block.get("type") == "image":
                content.append({"type": "image", "image": decode_image(block["image"])})
            else:
                content.append(block)
        prepared.append({"role": msg["role"], "content": content})
    return prepared


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, model_path
    logger.info(f"Loading model from {model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Model loaded.")
    yield
    del model, processor


app = FastAPI(title="Qwen3-VL Base Inference Server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = False


class GenerateResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int


@app.get("/health")
def health():
    return {"status": "ok", "model": model_path}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        messages = prepare_messages(req.messages)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": req.max_new_tokens,
            "do_sample": req.do_sample,
        }
        if req.do_sample:
            gen_kwargs["temperature"] = req.temperature

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][input_len:]
        response_text = processor.decode(new_ids, skip_special_tokens=True)

        return GenerateResponse(
            response=response_text,
            input_tokens=input_len,
            output_tokens=len(new_ids),
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)

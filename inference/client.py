"""
Client script for Qwen3-VL inference servers.

Usage examples:
    # Text only
    python inference/client.py --text "Navigate to the kitchen."

    # With image file
    python inference/client.py --image /path/to/scene.jpg --text "Which way should I go?"

    # Multi-turn conversation
    python inference/client.py --chat

    # Call SFT server (port 8001) instead of base (port 8000)
    python inference/client.py --port 8001 --image scene.jpg --text "Describe the scene."

    # Compare base vs SFT on the same input
    python inference/client.py --compare --image scene.jpg --text "Which direction?"
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests

BASE_URL_TEMPLATE = "http://{host}:{port}"


def encode_image(path: str) -> str:
    """Encode a local image file to a base64 data URI."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def build_messages(text: str, image: str | None) -> list[dict]:
    content = []
    if image:
        if Path(image).exists():
            content.append({"type": "image", "image": encode_image(image)})
        else:
            # treat as already a path the server can access directly
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def call_generate(base_url: str, messages: list[dict], max_new_tokens: int, do_sample: bool, temperature: float) -> dict:
    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
    }
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        info = resp.json()
        print(f"[Server] {base_url} — model: {info.get('model', '?')}")
        return True
    except Exception as e:
        print(f"[Error] Cannot reach {base_url}: {e}")
        return False


def run_single(args):
    base_url = BASE_URL_TEMPLATE.format(host=args.host, port=args.port)
    if not check_health(base_url):
        sys.exit(1)

    messages = build_messages(args.text, args.image)
    result = call_generate(base_url, messages, args.max_new_tokens, args.do_sample, args.temperature)
    print(f"\nResponse: {result['response']}")
    print(f"Tokens: {result['input_tokens']} in → {result['output_tokens']} out")


def run_compare(args):
    """Send the same prompt to both base (8000) and SFT (8001) and show side-by-side."""
    base_url_base = BASE_URL_TEMPLATE.format(host=args.host, port=8000)
    base_url_sft = BASE_URL_TEMPLATE.format(host=args.host, port=8001)

    ok_base = check_health(base_url_base)
    ok_sft = check_health(base_url_sft)
    if not (ok_base and ok_sft):
        sys.exit(1)

    messages = build_messages(args.text, args.image)

    print("\n--- Base Model ---")
    r_base = call_generate(base_url_base, messages, args.max_new_tokens, args.do_sample, args.temperature)
    print(r_base["response"])

    print("\n--- SFT Model ---")
    r_sft = call_generate(base_url_sft, messages, args.max_new_tokens, args.do_sample, args.temperature)
    print(r_sft["response"])


def run_chat(args):
    """Interactive multi-turn chat in the terminal."""
    base_url = BASE_URL_TEMPLATE.format(host=args.host, port=args.port)
    if not check_health(base_url):
        sys.exit(1)

    print("Multi-turn chat mode. Type 'quit' to exit, 'clear' to reset history.\n")
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[History cleared]\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
        result = call_generate(base_url, history, args.max_new_tokens, args.do_sample, args.temperature)
        reply = result["response"]
        history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"Model: {reply}\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL inference server client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001, help="Server port (base=8000, sft=8001)")
    parser.add_argument("--text", default="Describe what you see.")
    parser.add_argument("--image", default=None, help="Path to an image file")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chat", action="store_true", help="Interactive multi-turn chat")
    parser.add_argument("--compare", action="store_true", help="Compare base (8000) vs SFT (8001)")
    args = parser.parse_args()

    if args.compare:
        run_compare(args)
    elif args.chat:
        run_chat(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()

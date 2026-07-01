"""
One-time prep for the GRU-Qwen SFT base dir (chang).

Makes an existing alignment checkpoint dir (which has `model.safetensors` +
tokenizer with <gru>) loadable by `from_pretrained` by copying in the 4 config
files it lacks and fixing `vocab_size` to match the embedding rows.

This is a *one-time, additive* helper — it does NOT touch `model.safetensors`
or the tokenizer, and it is not part of the training framework. After running,
point training at the same dir:  MODEL_NAME_OR_PATH=$BASE_GRU_DIR

Usage:
    python tools/prepare_base_gru_dir_chang.py \
        --base_dir /home/rithvik/IROS_proj/models_ckpts/trained/gru/base_qwen_with_gru \
        --donor_dir /home/rithvik/IROS_proj/models_ckpts/downloaded/Qwen3-VL-8B/instruct

Add --force to overwrite config files that already exist, --dry-run to preview.
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

CONFIG_FILES = [
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
]
EMBED_KEYS = (
    "qwen.model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "qwen.lm_head.weight",
)
GRU_TOKEN = "<gru>"


def safetensors_embed_rows(path: Path):
    """Read the embedding row count (= true vocab size) from a .safetensors header."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for k in EMBED_KEYS:
        shape = header.get(k, {}).get("shape")
        if shape and len(shape) >= 2:
            return int(shape[0])
    return None


def patch_vocab_size(cfg: dict, vocab: int) -> list:
    """Set vocab_size to `vocab` at top level and under text_config if present."""
    changed = []
    if "vocab_size" in cfg and cfg["vocab_size"] != vocab:
        changed.append(f"vocab_size {cfg['vocab_size']} -> {vocab}")
        cfg["vocab_size"] = vocab
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        if tc.get("vocab_size") != vocab:
            changed.append(f"text_config.vocab_size {tc.get('vocab_size')} -> {vocab}")
            tc["vocab_size"] = vocab
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="alignment ckpt dir (has model.safetensors + tokenizer)")
    ap.add_argument("--donor_dir", required=True, help="a Qwen3-VL-8B dir to copy config files from")
    ap.add_argument("--set_architectures", action="store_true",
                    help="set config.architectures to the GRU class name")
    ap.add_argument("--force", action="store_true", help="overwrite config files that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir)
    donor = Path(args.donor_dir)
    if not base.is_dir():
        sys.exit(f"[ERR] base_dir not found: {base}")
    if not donor.is_dir():
        sys.exit(f"[ERR] donor_dir not found: {donor}")

    model_st = base / "model.safetensors"
    if not model_st.exists():
        sys.exit(f"[ERR] {model_st} missing (need the alignment full-model save)")

    vocab = safetensors_embed_rows(model_st)
    if vocab is None:
        sys.exit(f"[ERR] could not read embedding rows from {model_st}")
    print(f"[info] embedding rows in model.safetensors = {vocab}  (this becomes vocab_size)")

    # 1) copy the 4 config files (skip existing unless --force)
    for name in CONFIG_FILES:
        src, dst = donor / name, base / name
        if not src.exists():
            print(f"[warn] donor lacks {name}; skipping (verify the model still loads)")
            continue
        if dst.exists() and not args.force:
            print(f"[skip] {name} already in base_dir (use --force to overwrite)")
            continue
        print(f"[copy] {src} -> {dst}")
        if not args.dry_run:
            shutil.copy2(src, dst)

    # 2) patch config.json vocab_size (+ optional architectures).
    # In a real run config.json was just copied; in --dry-run read the donor to preview.
    cfg_path = base / "config.json"
    cfg_src = cfg_path if cfg_path.exists() else (donor / "config.json")
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text())
        changes = patch_vocab_size(cfg, vocab)
        if args.set_architectures and cfg.get("architectures") != ["Qwen3VLGRUForConditionalGeneration"]:
            changes.append("architectures -> ['Qwen3VLGRUForConditionalGeneration']")
            cfg["architectures"] = ["Qwen3VLGRUForConditionalGeneration"]
        print(f"[config.json] changes: {changes if changes else 'none'}")
        if changes and not args.dry_run:
            cfg_path.write_text(json.dumps(cfg, indent=2))
    else:
        print("[warn] no config.json in base_dir or donor")

    # 3) verify <gru> token
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(base), use_fast=False)
        gid = tok.convert_tokens_to_ids(GRU_TOKEN)
        print(f"[verify] {GRU_TOKEN} id = {gid}  ({'OK' if gid not in (None, tok.unk_token_id) else 'MISSING!'})")
    except Exception as e:
        print(f"[verify] tokenizer check skipped: {e}")

    print(f"\n[done] BASE_GRU_DIR ready{' (dry-run, nothing written)' if args.dry_run else ''}: {base}")
    print(f"       run:  BASE_GRU_DIR={base} bash scripts/slurm_gru_qwen_chang.sh")


if __name__ == "__main__":
    main()

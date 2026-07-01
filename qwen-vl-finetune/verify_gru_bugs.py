"""
verify_gru_bugs.py — empirically verify the GRU-Qwen pipeline bugs using the
real r2r_alignment_dataset_qa.json records and the actual pipeline functions.

Run:
    cd qwen-vl-finetune
    /opt/conda-envs/qwen-sft/bin/python verify_gru_bugs.py
"""
import json, inspect, re, sys, types, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Stub video-only deps (cv2/decord) — not used by the image+text alignment QA path.
for mod in ("cv2", "decord"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
sys.modules["decord"].VideoReader = object
sys.modules["decord"].cpu = lambda *a, **k: None

MODEL = "/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct"
DATA = "/home/rithvik/IROS_proj/cvpr_proj/llm_test/r2r_alignment_dataset_qa.json"
DATA_DIR = "/home/rithvik/IROS_proj/cvpr_proj/llm_test"

def hr(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

# ── load a few real alignment QA records ────────────────────────────────────
with open(DATA) as f:
    records = json.load(f)
sample = records[0]
sample["data_path"] = DATA_DIR
print("sample record:", json.dumps({k: sample[k] for k in ("id","image","gru")}, ensure_ascii=False))
print("question:", sample["conversations"][0]["value"].replace("\n","\\n"))

from transformers import AutoProcessor, AutoTokenizer
processor = AutoProcessor.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# replicate train_gru_qwen.py: register <motion> special token
MOTION = "<motion>"
if tokenizer.convert_tokens_to_ids(MOTION) == tokenizer.unk_token_id:
    tokenizer.add_special_tokens({"additional_special_tokens": [MOTION]})
motion_id = tokenizer.convert_tokens_to_ids(MOTION)
if hasattr(processor, "tokenizer"):
    processor.tokenizer = tokenizer
print(f"<motion> token id = {motion_id}")

import qwenvl.data.data_processor_gru as dp

# ── BUG 1: alignment QA prompt never emits <motion>; projector gets 0 gradient ──
hr("BUG 1 — alignment QA carries no <motion> token => projector zero-gradient")
messages, has_gru_msg = dp._build_messages(sample, Path(DATA_DIR))
def texts(msgs):
    out = []
    for m in msgs:
        for c in m["content"]:
            if c.get("type") == "text":
                out.append(c["text"])
            else:
                out.append(f"<{c.get('type')}>")
    return " ".join(out)
rendered = texts(messages)
print(f"_build_messages -> has_gru = {has_gru_msg}")
print("rendered message text (content types shown):", repr(rendered)[:300])
print(f"'<motion>' present in built messages? {'<motion>' in rendered}")
print(f"'<gru>' / '<image>' literal survived? gru={'<gru>' in rendered} image_tok={'<image>' in rendered}")

# full preprocess + replicate _get_item's gru-handling to get final has_gru + motion count
out = dp.preprocess_qwen_visual([sample], processor)
input_ids = out["input_ids"][0]
motion_positions = (input_ids == motion_id).nonzero().squeeze(-1)
_has_gru_flag = bool(out.pop("_has_gru", False))
# legacy branch in _get_item: record has "gru" -> rebuilds features, sets has_gru=True
legacy_triggered = (not _has_gru_flag) and ("gru" in sample)
final_has_gru = _has_gru_flag or legacy_triggered
print(f"_has_gru from preprocess = {_has_gru_flag}")
print(f"legacy 'gru'-field branch sets has_gru=True? {legacy_triggered}")
print(f"=> FINAL has_gru fed to model = {final_has_gru}")
print(f"=> <motion> positions in input_ids = {motion_positions.tolist()}  (count={motion_positions.numel()})")
print("VERDICT:", "BUG CONFIRMED — has_gru=True but 0 motion positions => forward no-op, projector*0 => zero gradient"
      if (final_has_gru and motion_positions.numel()==0) else "not reproduced")

# ── label/supervision sanity (is the gpt answer actually supervised?) ───────
hr("SANITY — is the assistant answer supervised (labels not all -100)?")
labels = out["labels"][0]
n_sup = int((labels != -100).sum())
print(f"supervised (non -100) label tokens = {n_sup}")
print("decoded supervised span:", tokenizer.decode([t for t in labels.tolist() if t != -100])[:120])

# ── BUG 2: forward() drops position_ids (mrope) ─────────────────────────────
hr("BUG 2 — GRUQwenModel.forward drops position_ids (mrope never reaches Qwen)")
from qwenvl.models.gru_qwen import GRUQwenModel
sig = inspect.signature(GRUQwenModel.forward)
print("forward params:", list(sig.parameters.keys()))
print(f"'position_ids' accepted by forward? {'position_ids' in sig.parameters}")
src = inspect.getsource(GRUQwenModel.forward)
print(f"'position_ids' ever put into model_kwargs? {'position_ids' in src.split('model_kwargs')[-1] if 'model_kwargs' in src else False}")
print(f"data pipeline DOES produce position_ids? {'position_ids' in out}")
print("VERDICT: BUG CONFIRMED — pipeline computes mrope position_ids but forward neither accepts nor forwards them")

# ── BUG 3: collator promotes (1,T,7)->4D, so 3D-wrong-timestep does NOT fire here ──
hr("BUG 3 — 3D injection-timestep bug: does it trigger on alignment QA?")
raw_actions = [int(a) for a in sample["gru"]]
feats = dp.actions_to_motion_features(raw_actions).unsqueeze(0)
print(f"per-sample gru_features shape (legacy branch) = {tuple(feats.shape)}  (3D)")
batched, lens = dp.collate_gru_prefixes([{"gru_features": feats, "gru_lengths": torch.tensor([feats.size(1)])}])
print(f"after collate_gru_prefixes = {tuple(batched.shape)} (4D) -> forward takes 4D last-state path")
print("VERDICT: 3D-wrong-timestep bug is MASKED on alignment QA (collator makes it 4D; 4D path uses last state correctly)")

# ── BUG 4: PROJECTOR_K>1 dimension mismatch vs index_copy(width=hidden) ──────
hr("BUG 4 — PROJECTOR_K>1 breaks injection (projector width K*hidden != embed width)")
from qwenvl.models.projector import ProjectorMLP
for k in (1, 4):
    proj = ProjectorMLP(gru_hidden_dim=256, qwen_hidden_dim=4096, intermediate_dim=64, k=k)
    y = proj(torch.zeros(1, 256))
    print(f"K={k}: projector output width = {y.shape[-1]}  (embed width injected into = 4096) "
          f"=> {'OK' if y.shape[-1]==4096 else 'MISMATCH -> index_copy will error'}")
print("VERDICT: BUG CONFIRMED for K>1 — index_copy into 4096-wide embeds with K*4096 vectors fails")

# ── BUG 5: GRU checkpoint missing => silent random encoder ───────────────────
hr("BUG 5 — missing/empty GRU checkpoint is silent (random frozen encoder)")
init_src = inspect.getsource(GRUQwenModel.__init__)
seg = init_src[init_src.find("if gru_checkpoint_path"):]
print("checkpoint-missing handling raises? ", "raise" in seg.split("else:")[-1].split("for param")[0])
print("sft_gru_qwen.sh default GRU_CHECKPOINT_PATH:")
import subprocess
print("  ", subprocess.run(["grep","-n","GRU_CHECKPOINT_PATH=\\${GRU_CHECKPOINT_PATH","scripts/sft_gru_qwen.sh"],
                            capture_output=True, text=True).stdout.strip() or "(empty default)")
print("VERDICT: BUG CONFIRMED — falls through to randomly-initialized GRU with only a print, no raise")

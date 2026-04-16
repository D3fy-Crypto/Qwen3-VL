# GRU-Qwen Finetuning Implementation Summary

## Files Created

### 1. **Projector Module**
**File**: `qwenvl/models/projector.py`
- Simple MLP: 256 → 1024 → 4096 (or K*4096)
- Used to project GRU hidden states to Qwen embedding dimension
- Trainable by default

### 2. **GRU-Qwen Combined Model**
**File**: `qwenvl/models/gru_qwen.py`
- Wraps projector + Qwen model
- Handles forward pass: gru_hidden → projector → concat with text → Qwen
- Configurable which modules to train (projector, Qwen LM, Qwen vision)
- Outputs `GRUQwenOutput` with loss and logits

### 3. **Training Script**
**File**: `qwenvl/train/train_gru_qwen.py`
- Based on original `train_qwen.py`
- Initializes GRU-Qwen model instead of vanilla Qwen
- Supports gradient checkpointing, LoRA, distributed training
- Handles model checkpoint saving

### 4. **Training Shell Script**
**File**: `scripts/sft_gru_qwen.sh`
- Bash wrapper for training
- Sets hyperparameters (lr=1e-4, batch_size=4, etc.)
- Launches distributed training with torchrun

### 5. **Data Format Guide**
**File**: `DATA_FORMAT_GRU_QWEN.md`
- Instructions for preparing trajectory + text datasets
- Example data structures (raw trajectories vs pre-computed GRU hidden)
- Custom dataset class template
- Pre-computation scripts

### 6. **Documentation**
**File**: `FINETUNING_GUIDE.md` (created in llm_test)
- Overview of GRU-Qwen architecture
- Explanation of three training modes (projector-only, projector+LM, full)
- Integration steps and next steps

---

## Key Design Decisions

### Training Modes (configure in sft_gru_qwen.sh)

```bash
# Mode 1: Train Projector Only (RECOMMENDED - start here)
--tune_projector True
--tune_qwen_vision False
--tune_qwen_lm False
--lr 1e-4

# Mode 2: Train Projector + Qwen LM Head
--tune_projector True
--tune_qwen_vision False
--tune_qwen_lm True
--lr 5e-5

# Mode 3: Full Fine-tuning with LoRA (memory-intensive)
--lora_enable True
--lora_r 64
--tune_projector True
--tune_qwen_lm True
--lr 1e-4
```

### Default Configuration (sft_gru_qwen.sh)
```
GRU Checkpoint: /home/rithvik/IROS_proj/cvpr_proj/traj_model/checkpoints/best_model.pt
Qwen Model: Qwen/Qwen2.5-VL-7B-Instruct
Batch Size: 4
Learning Rate: 1e-4 (projector only)
Epochs: 3
Output: ./output_gru_qwen/
```

---

## What Still Needs to be Done

### 1. **Data Loading Integration** (CRITICAL)
**File to modify**: `qwenvl/data/data_processor.py`

Add trajectory dataset support:
```python
def make_supervised_data_module(processor, data_args):
    # ... existing code ...
    
    if "trajectory" in data_args.dataset_use:
        # Load from TrajectoryQADataset
        # Return train_dataset, eval_dataset, data_collator
```

### 2. **Create Trajectory Dataset Class** (CRITICAL)
**File to create**: `qwenvl/data/trajectory_dataset.py`

Template provided in `DATA_FORMAT_GRU_QWEN.md`

### 3. **Pre-compute GRU Hidden States** (OPTIONAL but RECOMMENDED)
Create: `scripts/precompute_gru_hidden.py`

Avoids re-running GRU during training (speeds up 10x)

### 4. **Prepare Your Dataset**
- Collect trajectory sequences (navigation paths, waypoints, etc.)
- Pair with text queries/instructions
- Format as JSON/JSONL
- Pre-compute GRU hidden states

### 5. **Update Training Arguments** (OPTIONAL)
**File**: `qwenvl/train/argument.py`

Add:
```python
gru_checkpoint_path: str = None
projector_k: int = 1
tune_projector: bool = True
tune_qwen_vision: bool = False
tune_qwen_lm: bool = False
```

---

## Quick Start

### Step 1: Verify File Creation
```bash
cd /home/rithvik/IROS_proj/cvpr_proj/Qwen3-VL/qwen-vl-finetune

# Should exist:
ls qwenvl/models/projector.py
ls qwenvl/models/gru_qwen.py
ls qwenvl/train/train_gru_qwen.py
ls scripts/sft_gru_qwen.sh
```

### Step 2: Create Data Class
Copy template from `DATA_FORMAT_GRU_QWEN.md` to `qwenvl/data/trajectory_dataset.py`

### Step 3: Prepare Dataset
Create trajectory + text JSON dataset, optionally pre-compute GRU hidden states

### Step 4: Run Training
```bash
bash scripts/sft_gru_qwen.sh
```

Or with torchrun directly:
```bash
torchrun --nproc_per_node=1 \
    qwenvl/train/train_gru_qwen.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gru_checkpoint_path /path/to/best_model.pt \
    --dataset_use trajectory_qa \
    --output_dir ./output \
    --tune_projector True
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GRU-Qwen Training Pipeline               │
└─────────────────────────────────────────────────────────────┘

INPUT:
├─ Trajectory sequence      [batch, seq_len, feat_dim]
├─ Text query              [batch, text_len]
└─ Response labels         [batch, response_len]

PROCESSING:
├─ GRU Encoder (FROZEN, from checkpoint)
│  └─ Output: [batch, seq_len, 256]
│
├─ Projector MLP (TRAINABLE)
│  └─ Linear(256→1024) + GELU + Linear(1024→4096)
│  └─ Output: [batch, seq_len, 4096]
│
├─ Qwen Tokenizer
│  └─ Text tokens: [batch, text_len]
│
├─ Concatenation
│  └─ [traj_projected + text_embeddings, attention_masks]
│  └─ Output: [batch, seq_len+text_len, 4096]
│
└─ Qwen Language Model (PARTIALLY TRAINABLE or FROZEN)
   ├─ Vision encoder (optional, frozen by default)
   ├─ Merger/connector (optional)
   └─ Language model (trainable if tune_qwen_lm=True)

OUTPUT:
├─ Logits:  [batch, seq_len+text_len, vocab_size]
├─ Loss:    scalar (next-token prediction loss)
└─ Gradients: update projector weights

OPTIMIZATION:
├─ Optimizer: AdamW
├─ Learning rate: 1e-4 (default, adjustable)
├─ Gradient accumulation: 4 steps
└─ Checkpointing: Every 500 steps
```

---

## Hyperparameter Recommendations

| Parameter | Projector-Only | Projector+LM | Full FT (LoRA) |
|-----------|---|---|---|
| Learning Rate | 1e-4 | 5e-5 | 1e-4 |
| Batch Size | 4-8 | 2-4 | 2-4 |
| Gradient Accum | 4 | 8 | 8 |
| Warmup Ratio | 0.05 | 0.1 | 0.1 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Epochs | 3 | 5 | 5 |
| LoRA Rank | N/A | N/A | 64 |
| Gradient Checkpoint | Yes | Yes | Yes |

---

## Expected Output Structure

After training, your `output_gru_qwen/` will contain:
```
output_gru_qwen/
├── checkpoint-500/
│   ├── pytorch_model.bin      (full model including projector)
│   ├── adapter_config.json    (if using LoRA)
│   └── adapter_model.bin      (if using LoRA)
├── checkpoint-1000/
├── ...
├── pytorch_model.bin          (final model)
├── tokenizer.model
├── tokenizer.json
├── config.json
└── training_args.bin
```

---

## Validation & Testing

After training, to use the fine-tuned model:

```python
from qwenvl.models.gru_qwen import GRUQwenModel
import torch

# Load fine-tuned model
model = GRUQwenModel(
    qwen_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    gru_checkpoint_path="/path/to/best_model.pt",
    device="cuda"
)

# Load fine-tuned weights
state_dict = torch.load("output_gru_qwen/pytorch_model.bin")
model.load_state_dict(state_dict)
model.eval()

# Inference
gru_hidden = torch.randn(1, 10, 256, device="cuda")  # Example trajectory
input_ids = torch.tensor([[101, 2023, ...]], device="cuda")  # Example text

with torch.no_grad():
    outputs = model(
        gru_hidden=gru_hidden,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids)
    )
    logits = outputs.logits
```

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `qwenvl/models/projector.py` | MLP projector | ✅ Created |
| `qwenvl/models/gru_qwen.py` | Combined model | ✅ Created |
| `qwenvl/train/train_gru_qwen.py` | Training script | ✅ Created |
| `scripts/sft_gru_qwen.sh` | Training launcher | ✅ Created |
| `DATA_FORMAT_GRU_QWEN.md` | Data prep guide | ✅ Created |
| `FINETUNING_GUIDE.md` | Architecture guide | ✅ Created |
| `qwenvl/data/trajectory_dataset.py` | Data class (TEMPLATE) | 🔲 Create from template |
| `qwenvl/data/data_processor.py` | Data processor (UPDATE) | 🔲 Needs modification |
| `qwenvl/train/argument.py` | Training args (UPDATE) | 🔲 Optional update |
| `scripts/precompute_gru_hidden.py` | GRU pre-computation | 🔲 Optional helper script |

---

## Troubleshooting

### Q: "Unrecognized configuration class Qwen3VLConfig"
**A**: The training script handles this automatically with fallback model loading.

### Q: "CUDA OOM error"
**A**: Reduce `per_device_train_batch_size` (try 2 instead of 4) or enable LoRA

### Q: "gru_hidden has wrong shape"
**A**: Verify your dataset returns shape `(batch, seq_len, 256)` not `(batch, 256)` or `(seq_len, 256)`

### Q: "Projector gradients are None"
**A**: Check that `tune_projector=True` in sft_gru_qwen.sh

### Q: "Loss doesn't decrease"
**A**: Try adjusting learning rate (start with 1e-4, adjust up/down), check batch normalization

---

## Next: Integration Checklist

- [ ] Verify all new files created in qwenvl/
- [ ] Create `qwenvl/data/trajectory_dataset.py` from template
- [ ] Update `qwenvl/data/data_processor.py` to load trajectory data
- [ ] Prepare trajectory dataset (raw + text) in JSON format
- [ ] (Optional) Pre-compute GRU hidden states
- [ ] Run training: `bash scripts/sft_gru_qwen.sh`
- [ ] Monitor training on WandB dashboard
- [ ] Evaluate fine-tuned model on validation set

# GRU-Qwen Data Format & Setup

## Input Data Format

Your training data needs trajectory sequences + corresponding text queries/instructions.

### Option A: Trajectory Sequence + Text (Recommended)

```json
{
    "trajectory": [
        [x1, y1, z1, ...],  // t=0
        [x2, y2, z2, ...],  // t=1
        ...
        [xn, yn, zn, ...]   // t=n
    ],
    "text": "Navigate to the kitchen and find a cup",
    "response": "I found the cup in the kitchen cabinet"
}
```

### Option B: Pre-computed GRU Hidden States (Faster)

```json
{
    "gru_hidden": [
        shape: (seq_len, 256),
        values: [...]
    ],
    "text": "Navigate to the kitchen",
    "response": "Found the cup"
}
```

---

## Data Processing Flow

### Current Pipeline (change nothing):
```
Raw Trajectory Data
    ↓ [Load Trajectory]
    ↓ [GRU Encoder] 
    ↓ Optional: Cache GRU hidden states
    ↓ [Projector] 256 → 4096
    ↓ [Concat with text embeddings]
    ↓ [Qwen] → logits/loss
```

---

## Minimal Trajectory Dataset Example

Create a custom dataset class in `qwenvl/data/`:

```python
# qwenvl/data/trajectory_dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class TrajectoryQADataset(Dataset):
    def __init__(self, data_file: str, tokenizer, gru_model=None):
        """
        Args:
            data_file: JSON file with trajectory + text pairs
            tokenizer: Tokenizer for text
            gru_model: Optional pre-loaded GRU model for on-the-fly encoding
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.gru_model = gru_model
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get GRU hidden states
        if "gru_hidden" in item:
            # Pre-computed
            gru_hidden = torch.tensor(item["gru_hidden"], dtype=torch.float32)
        else:
            # Compute on-the-fly (requires GRU model)
            trajectory = torch.tensor(item["trajectory"], dtype=torch.float32)
            if self.gru_model is not None:
                with torch.no_grad():
                    gru_hidden = self.gru_model(trajectory.unsqueeze(0))
            else:
                raise ValueError("gru_model required if gru_hidden not pre-computed")
        
        # Tokenize text
        text = item.get("text", "")
        text_encoding = self.tokenizer(text, return_tensors="pt", max_length=512)
        
        # Tokenize response for labels
        response = item.get("response", "")
        response_encoding = self.tokenizer(response, return_tensors="pt", max_length=512)
        
        return {
            "gru_hidden": gru_hidden,
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "labels": response_encoding["input_ids"].squeeze(),
        }
```

---

## How to Prepare Your Trajectory Data

### Step 1: Collect Trajectories
Format: Each trajectory is a sequence of states/waypoints

```python
trajectories = [
    {
        "trajectory": [[x1, y1, z1], [x2, y2, z2], ...],  # shape: (seq_len, 3)
        "text": "Go to the door",
        "response": "I'm going to the door"
    },
    ...
]
```

### Step 2: Pre-compute GRU Hidden States (Optional but Recommended)
Avoids re-running GRU during training.

```python
import torch
from pathlib import Path
import json

# Load pre-trained GRU
gru_ckpt = torch.load("/path/to/best_model.pt")
# Assume GRU model is in gru_ckpt["model_state_dict"]

# For each trajectory
dataset_with_gru_hidden = []
for item in trajectories:
    trajectory_tensor = torch.tensor(item["trajectory"], dtype=torch.float32)
    
    # Forward through GRU
    with torch.no_grad():
        gru_h = gru_model(trajectory_tensor)  # (seq_len, 256)
    
    item["gru_hidden"] = gru_h.cpu().tolist()
    dataset_with_gru_hidden.append(item)

# Save for training
with open("trajectory_qa_dataset.json", "w") as f:
    json.dump(dataset_with_gru_hidden, f)
```

### Step 3: Create JSONL Format (HuggingFace compatible)

```json
{"trajectory": [...], "gru_hidden": [...], "text": "...", "response": "..."}
{"trajectory": [...], "gru_hidden": [...], "text": "...", "response": "..."}
...
```

---

## Training with Trajectory Data

### Modify `qwenvl/data/data_processor.py`

Add trajectory dataset support:

```python
def make_supervised_data_module(processor, data_args):
    """Create training data module."""
    
    if "trajectory" in data_args.dataset_use.lower():
        from qwenvl.data.trajectory_dataset import TrajectoryQADataset
        
        train_dataset = TrajectoryQADataset(
            data_file=data_args.data_path,
            tokenizer=processor
        )
        eval_dataset = None  # Optional
        
        # Collate function for batching
        def collate_fn(batch):
            gru_hidden = torch.stack([item["gru_hidden"] for item in batch])
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            
            return {
                "gru_hidden": gru_hidden,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": collate_fn,
        }
    
    # ... rest of original code
```

---

## Example Training Command

```bash
# 1. Pre-compute GRU hidden states (optional)
python scripts/precompute_gru_hidden.py \
    --raw_trajectories data/trajectories.json \
    --gru_checkpoint traj_model/checkpoints/best_model.pt \
    --output data/trajectories_with_gru.json

# 2. Run training
bash scripts/sft_gru_qwen.sh
```

---

## Configuration in sft_gru_qwen.sh

```bash
# Set your data paths
datasets=trajectory_qa  # Dataset name in qwenvl/data/data_processor.py

# Adjust hyperparameters
lr=1e-4              # Learning rate for projector
batch_size=4         # Batch size
num_train_epochs=3   # Number of training epochs
```

---

## What Happens During Training

For each mini-batch:

```
Input: gru_hidden (batch, seq_len, 256)
    ↓
Projector: (batch, seq_len, 256) → (batch, seq_len, 4096)
    ↓
Concatenate: [projected_traj, text_embeddings] → (batch, seq_len_total, 4096)
    ↓
Qwen LM: process combined embeddings
    ↓
Loss: compute next-token prediction loss on labels
    ↓
Backprop: only through projector (other modules frozen)
    ↓
Optimizer.step() on projector.parameters()
```

---

## Validation Metrics

Track during training:
- Loss (next-token prediction)
- Training time per step
- Projector gradient norms

Example:
```bash
# From WandB logs:
train/loss: 2.34
train/learning_rate: 1e-4
train/projector.net.0.weight_grad_norm: 0.45
```

---

## Next Steps

1. **Prepare data**: Collect trajectory sequences + text pairs
2. **Pre-compute GRU hidden**: Optional but recommended for speed
3. **Update data_processor.py**: Add trajectory dataset support
4. **Run training**: `bash scripts/sft_gru_qwen.sh`
5. **Monitor**: Check WandB dashboard for losses and metrics

Questions? Check the files:
- Train script: `qwenvl/train/train_gru_qwen.py`
- Model: `qwenvl/models/gru_qwen.py`
- Projector: `qwenvl/models/projector.py`

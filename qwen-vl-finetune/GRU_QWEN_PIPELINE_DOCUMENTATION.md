# GRU-Qwen Pipeline Documentation

This document records the exact GRU-Qwen finetuning path in this workspace, the dataset contract, the tensor dimensions at each block, and the smoke-test launch command used for the `max_steps=1` run.

## 1. What This Pipeline Does

The pipeline fine-tunes Qwen on navigation-style examples where the prompt includes a trajectory summary marker (`<gru>`) plus an image slot (`<image>`). The raw GRU actions are converted into fixed 7D motion features, encoded by a frozen GRU, projected into Qwen token space, concatenated with text embeddings, and then consumed by the Qwen causal language model.

The current implementation keeps the GRU encoder frozen and trains the projector by default. Qwen vision and Qwen language weights stay frozen unless you explicitly enable them.

## 2. Dataset Audit

Two dataset files are present in `llm_test`:

- `llm_test/r2r_alignment_dataset_qa.json`
- `llm_test/r2r_alignment_dataset_qa (1).json`

They share the same schema and record count, but they are not byte-identical.

Measured stats for both files:

- Record count: 353,894
- Required keys present in the current pipeline: `id`, `image`, `gru`, `conversations`
- Conversation structure: exactly 2 turns in every inspected record
- GRU action length: min 0, max 288, mean 15.18, median 13
- Unique image paths: 288,594
- Missing image files under `llm_test/`: 288,594 of 288,594 unique paths

The loader handles missing image files by replacing the image placeholder with the text token `[missing_image]`, so the current smoke run can still proceed as text-plus-GRU training even though the image files are absent from the workspace.

Recommended canonical file for training in this workspace:

- `llm_test/r2r_alignment_dataset_qa.json`

Reason: it is the primary-named dataset, it matches the current loader, and the companion copy only adds ambiguity without changing the schema.

### Example dataset record

```json
{
  "id": "r2r_914-23",
  "image": ["914/frame_48.jpg"],
  "gru": [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<gru>\nWhat motion concludes this sequence?"
    },
    {
      "from": "gpt",
      "value": "The sequence ends with the robot moving forward 25cm."
    }
  ]
}
```

## 3. Dataset Processing Flow

The current data path is implemented in `qwenvl/data/data_processor.py` and the dataset registry in `qwenvl/data/__init__.py`.

### 3.1 Dataset registry

`qwenvl/data/__init__.py` registers:

- `r2r_alignment_qa`

This maps to:

- `annotation_path = /home/rithvik/IROS_proj/cvpr_proj/llm_test/r2r_alignment_dataset_qa.json`
- `data_path = /home/rithvik/IROS_proj/cvpr_proj/llm_test`

### 3.2 Loader behavior

`LazySupervisedDataset` performs the following steps for each sample:

1. Load the JSON list from the annotation file.
2. Attach `data_path` to each example.
3. Call `preprocess_qwen_visual` to build a Qwen-style conversation.
4. Tokenize the conversation with `processor.apply_chat_template(...)`.
5. Extract the raw trajectory action IDs from the `gru` field.
6. Convert raw actions into 7D motion features with `actions_to_motion_features(...)`.
7. Attach `gru_features` and `gru_length` to the returned sample.

### 3.3 Raw action to motion feature conversion

The action vocabulary is:

- `0 = STOP`
- `1 = FORWARD`
- `2 = TURN_LEFT`
- `3 = TURN_RIGHT`

`actions_to_motion_features` turns the integer action sequence into one row per timestep with the following feature order:

1. cumulative world x
2. cumulative world y
3. `sin(theta)`
4. `cos(theta)`
5. `dyaw`
6. `is_forward`
7. `is_turn`

So the GRU input tensor is always 7-dimensional per timestep.

### 3.4 Missing image behavior

If the sample references an image path that does not exist, the loader substitutes `[missing_image]` in the prompt content instead of failing. That is why the dataset can still be used in this workspace even though the image tree is absent.

## 4. Batch Construction

The collator in `qwenvl/data/data_processor.py` merges samples into a batch with these tensors:

- `input_ids`: padded text tokens
- `labels`: padded causal-LM labels, with prefix positions masked to `-100`
- `attention_mask`: text attention mask
- `position_ids`: Qwen RoPE position ids
- `pixel_values`, `image_grid_thw`: present only if real images were loaded
- `pixel_values_videos`, `video_grid_thw`: present only if video inputs were loaded
- `gru_features`: padded GRU feature tensor, shape `[B, T_max, 7]`
- `gru_lengths`: original lengths, shape `[B]`

## 5. Model Architecture

The model class is `qwenvl/models/gru_qwen.py`.

### 5.1 GRU encoder

The notebook-compatible encoder is:

- `input_dim = 7`
- `hidden_dim = 256`
- `embedding_dim = 128`

Important note: the `embedding_dim` in the notebook is part of the GRU-side head, but the training pipeline here uses the GRU sequence output directly. The sequence output shape is `[B, T, 256]`.

### 5.2 Projector

`qwenvl/models/projector.py` uses:

- `Linear(256 -> 1024)`
- `GELU`
- `Linear(1024 -> 4096)`

The local Qwen config in `qwen_models/instruct/config.json` reports `text_config.hidden_size = 4096`, so the projector output dimension matches the Qwen token embedding width.

### 5.3 Qwen model

The local model cache at `qwen_models/instruct` reports:

- `model_type = qwen3_vl`
- `text_config.hidden_size = 4096`
- `vision_config.hidden_size = 1152`

That means the smoke test can run from the local cache without hitting the Hugging Face network, and the trainer should infer `qwen3vl` from config rather than from the path string.

## 6. Forward Pass Dimensions

The forward path in `GRUQwenModel.forward(...)` is:

1. Input `gru_features`: `[B, T_gru, 7]`
2. Input `gru_lengths`: `[B]`
3. GRU encoder output: `[B, T_gru, 256]`
4. Projector output: `[B, T_gru, 4096]`
5. Qwen text embeddings: `[B, T_text, 4096]`
6. Concatenated embeddings: `[B, T_gru + T_text, 4096]`
7. Concatenated attention mask: `[B, T_gru + T_text]`
8. Prefix labels masked to `-100`: `[B, T_gru]`
9. Combined labels: `[B, T_gru + T_text]`
10. Qwen logits: `[B, T_gru + T_text, vocab_size]`

### Example forward with concrete shapes

Use this as the shape reference for a small synthetic batch:

- `B = 2`
- `T_gru = 23`
- `T_text = 48`
- `gru_features = [2, 23, 7]`
- `gru_lengths = [2]`
- `gru_hidden = [2, 23, 256]`
- `projected = [2, 23, 4096]`
- `text_embeds = [2, 48, 4096]`
- `combined_embeds = [2, 71, 4096]`
- `combined_attention = [2, 71]`
- `combined_labels = [2, 71]`

The first 23 tokens are the GRU prefix; the remaining 48 tokens are the text side.

## 7. Example Forward Trace

The intended debug print format from the forward pass is:

```text
[GRU-Qwen][debug] batch gru_features=(2, 23, 7) gru_hidden=(2, 23, 256) projected=(2, 23, 4096)
[GRU-Qwen][debug] logits shape=(2, 71, vocab_size)
```

If you want to reproduce that outside the full trainer, use the same tensor dimensions above and step through:

1. GRU encoder
2. Projector
3. `torch.cat([projected, text_embeds], dim=1)`
4. Prefix label masking

## 8. Smoke-Test Launch

The shell launcher has been updated to support environment overrides. For the requested one-step real training pass, use:

```bash
cd /home/rithvik/IROS_proj/cvpr_proj/Qwen3-VL/qwen-vl-finetune

MAX_STEPS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
NUM_TRAIN_EPOCHS=1 \
MODEL_MAX_LENGTH=512 \
SAVE_STEPS=1 \
LOGGING_STEPS=1 \
REPORT_TO=none \
USE_DEEPSPEED=0 \
DATASETS=r2r_alignment_qa \
MODEL_NAME_OR_PATH=/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct \
bash scripts/sft_gru_qwen.sh
```

Why these overrides matter:

- `MAX_STEPS=1` forces a true one-step smoke run.
- `PER_DEVICE_TRAIN_BATCH_SIZE=1` reduces memory pressure.
- `MODEL_NAME_OR_PATH` points at the local cached Qwen3-VL model instead of requiring a new download.
- `REPORT_TO=none` avoids W&B setup during the smoke test.

## 9.1 Validated Smoke Run Output (April 11, 2026)

The run was executed in GPU environment `vla-env` with:

- `python`: `/opt/conda-envs/vla-env/bin/python`
- `torch`: `2.11.0+cu130`
- launcher: `scripts/sft_gru_qwen.sh`
- overrides: `MAX_STEPS=1`, `PER_DEVICE_TRAIN_BATCH_SIZE=1`, `USE_DEEPSPEED=0`

Key observed logs from the successful pass:

- `[GRU-Qwen] Dataset size: 353894`
- `[GRU-Qwen][debug] batch gru_features=(1, 23, 7) gru_hidden=(1, 23, 256) projected=(1, 23, 4096)`
- `[GRU-Qwen][debug] logits shape=(1, 69, 151936)`
- `{'loss': '3.077', 'grad_norm': '73.21', 'learning_rate': '0', 'epoch': '2.826e-06'}`
- `{'train_runtime': '18.44', 'train_samples_per_second': '0.054', 'train_steps_per_second': '0.054', 'train_loss': '3.077', 'epoch': '2.826e-06'}`

Artifacts observed in `output_gru_qwen_smoke/`:

- `checkpoint-1/model.safetensors`
- `checkpoint-1/optimizer.pt`
- `checkpoint-1/scheduler.pt`
- `model.safetensors`
- `trainer_state.json`

## 9. What To Expect During The Run

Expected log sequence:

1. Launcher banner with model, dataset, output directory, and step budget.
2. Trainer startup.
3. Qwen config inference resolving `qwen3_vl -> qwen3vl`.
4. Dataset loading from `r2r_alignment_dataset_qa.json`.
5. Collator batch creation including `gru_features` and `gru_lengths`.
6. One optimizer step.
7. Checkpoint write into `output_gru_qwen_smoke/`.

If the GRU checkpoint path is absent, the model now reports that it is using the randomly initialized trajectory GRU instead of silently failing.

## 10. Implementation Files

Relevant files in this workspace:

- `Qwen3-VL/qwen-vl-finetune/scripts/sft_gru_qwen.sh`
- `Qwen3-VL/qwen-vl-finetune/qwenvl/train/train_gru_qwen.py`
- `Qwen3-VL/qwen-vl-finetune/qwenvl/models/gru_qwen.py`
- `Qwen3-VL/qwen-vl-finetune/qwenvl/models/projector.py`
- `Qwen3-VL/qwen-vl-finetune/qwenvl/data/data_processor.py`
- `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`

import sys
sys.path.insert(0, "/weka/home/djonna1/cvpr_proj/Qwen3-VL/qwen-vl-finetune")

import torch
from types import SimpleNamespace
from transformers import AutoProcessor
from qwenvl.data.data_processor import LazySupervisedDataset

MODEL_PATH = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Model/instruct"

data_args = SimpleNamespace(
    dataset_use="r2r,human,rxr,scanqa",
    max_pixels=50176,
    min_pixels=784,
    video_min_pixels=784,
    video_max_pixels=50176,
    video_min_frames=4,
    video_max_frames=32,
    video_fps=1.0,
    model_max_length=4096,
    model_type="qwen3vl",
    data_flatten=True,
    data_packing=False,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
dataset = LazySupervisedDataset(processor, data_args=data_args)

sample = dataset[0]
total_bytes = sum(v.nbytes for v in sample.values() if isinstance(v, torch.Tensor))
print(f"Sample size S = {total_bytes / 1e6:.1f} MB")
print(f"  Tensors: { {k: tuple(v.shape) for k, v in sample.items() if isinstance(v, torch.Tensor)} }")

# Plug into formula: RAM = 14 + N_gpu * N_worker * (1.5 + prefetch * B * S_gb) + 3
N_gpu, N_worker, prefetch, B = 4, 8, 2, 4
S_gb = total_bytes / 1e9
ram = 14 + N_gpu * N_worker * (1.5 + prefetch * B * S_gb) + 3
print(f"\nEstimated RAM = 14 + {N_gpu}×{N_worker}×(1.5 + {prefetch}×{B}×{S_gb:.3f}) + 3 = {ram:.1f} GB")
print(f"Recommended --mem = {int(ram * 1.2 / 32 + 1) * 32}G  (×1.2 safety margin, rounded to 32G)")

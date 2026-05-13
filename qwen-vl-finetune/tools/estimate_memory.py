from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

MODEL_PATH = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Model/instruct"

model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)

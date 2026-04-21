from transformers import AutoProcessor, AutoTokenizer

from qwenvl.train.argument import DataArguments
from qwenvl.data.data_processor_gru import make_supervised_data_module


MODEL_PATH = "/home/rithvik/IROS_proj/cvpr_proj/qwen_models/instruct"
DATASETS = ["r2r", "rxr", "human"]


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    motion_token_text = "<motion>"
    if tokenizer.convert_tokens_to_ids(motion_token_text) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [motion_token_text]})
    return tokenizer


def main() -> None:
    for dataset_name in DATASETS:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        tokenizer = build_tokenizer()
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer

        data_args = DataArguments(
            dataset_use=dataset_name,
            model_type="qwen3vl",
            max_pixels=50176,
            min_pixels=784,
            gru_history_slots=8,
            motion_token_text="<motion>",
        )

        data_module = make_supervised_data_module(processor, data_args=data_args)
        sample = data_module["train_dataset"][0]
        batch = data_module["data_collator"]([sample])

        input_ids = batch["input_ids"][0]
        motion_token_id = int(batch.get("motion_token_id", tokenizer.convert_tokens_to_ids("<motion>")))
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))

        motion_positions = (input_ids == motion_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        collator_motion_positions = batch.get("motion_positions", [[]])[0]

        print(f"=== {dataset_name} ===")
        print(f"motion_token_id={motion_token_id} image_token_id={image_token_id}")
        print(f"motion_count={len(motion_positions)} image_count={len(image_positions)}")
        print(f"motion_positions_match_batch={list(motion_positions) == list(collator_motion_positions)}")
        print(f"motion_positions={motion_positions}")
        print(f"image_positions={image_positions[:16]}{' ...' if len(image_positions) > 16 else ''}")


if __name__ == "__main__":
    main()
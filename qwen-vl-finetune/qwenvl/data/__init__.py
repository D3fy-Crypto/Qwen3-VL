import re
from pathlib import Path

_LOCAL_NAVILA_BASE = Path("/home/rithvik/IROS_proj/NaVILA-Dataset")
if _LOCAL_NAVILA_BASE.exists():
    NAVILA_BASE = str(_LOCAL_NAVILA_BASE)
else:
    NAVILA_BASE = "/weka/scratch/tinoosh/iros_dataset/NaVILA-Dataset"


def _first_existing_path(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return str(Path(candidates[0]))

R2R = {
    "annotation_path": f"{NAVILA_BASE}/R2R/annotations.json",
    "data_path": f"{NAVILA_BASE}/R2R/train",
}

ENVDROP = {
    "annotation_path": f"{NAVILA_BASE}/EnvDrop/annotations.json",
    "data_path": f"{NAVILA_BASE}/EnvDrop/train",
}

HUMAN = {
    "annotation_path": f"{NAVILA_BASE}/Human/annotations.json",
    "data_path": f"{NAVILA_BASE}/Human/raw_frames",
}

RXR = {
    "annotation_path": f"{NAVILA_BASE}/RxR/annotations.json",
    "data_path": f"{NAVILA_BASE}/RxR/train",
}

SCANQA = {
    "annotation_path": f"{NAVILA_BASE}/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
    "data_path": f"{NAVILA_BASE}/ScanQA/videos",
}

_R2R_ALIGNMENT_QA_ANN = _first_existing_path(
    [
        "/home/rithvik/IROS_proj/cvpr_proj/llm_test/r2r_alignment_dataset_qa.json",
        "/home/rithvik/IROS_proj/llm_test/r2r_alignment_dataset_qa.json",
    ]
)
_R2R_ALIGNMENT_QA_DATA = _first_existing_path(
    [
        "/home/rithvik/IROS_proj/cvpr_proj/llm_test",
        "/home/rithvik/IROS_proj/llm_test",
    ]
)

R2R_ALIGNMENT_QA = {
    "annotation_path": _R2R_ALIGNMENT_QA_ANN,
    "data_path": _R2R_ALIGNMENT_QA_DATA,
}

data_dict = {
    "r2r": R2R,
    "envdrop": ENVDROP,
    "human": HUMAN,
    "rxr": RXR,
    "scanqa": SCANQA,
    "r2r_alignment_qa": R2R_ALIGNMENT_QA,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["r2r"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)

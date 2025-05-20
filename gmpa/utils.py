from typing import Any, Literal, Optional
import os
from loguru import logger


def get_device() -> str:
    """
    Get the device to be used for training or inference.
    This function checks for the availability of NPUs or GPUs and sets the device accordingly.
    If neither is available, it defaults to CPU.
    """
    import torch

    try:
        import torch_npu
        from _flash_attn_npu import patch_npu_flash_attn

    except Exception as e:
        print(e)

    device = None
    if hasattr(torch, "npu"):
        if torch.npu.is_available():
            num_gpus = torch.npu.device_count()
            logger.info(f"Available NPUs: {num_gpus}")
            device = "npu:0"
    elif hasattr(torch, "cuda"):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available CUDAs: {num_gpus}")
            device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")


def save(
    obj: Any,
    path: str,
    name: Optional[str] = None,
    type: Optional[Literal["pkl", "jsonl", "faiss", "npy"]] = None,
):

    if name is None and path.find(".") != -1:
        name, type = path.split(".")
        name = name.split("/")[-1]
        path = path[: path.rfind("/") + 1]

    match type:
        case "pkl":
            import pickle

            with open(f"{path}{name}.pkl", "wb") as f:
                pickle.dump(obj, f)
                print(f"Saved {name} to {path}")
        case "jsonl":
            import jsonlines

            with jsonlines.open(f"{path}{name}.jsonl", "w") as writer:
                writer.write_all(obj)
                print(f"Saved {name} to {path}")
        case "faiss":
            import faiss

            faiss.write_index(obj, f"{path}{name}.faiss")
            print(f"Saved {name} to {path}")
        case "npy":
            import numpy as np

            np.save(f"{path}{name}.npy", obj)
            print(f"Saved {name} to {path}")


def load(
    path: str,
    name: Optional[str] = None,
    type: Optional[Literal["pkl", "jsonl", "faiss", "txt", "npy"]] = None,
):

    if path.find(".") != -1:
        name, type = path.split(".")
        name = name.split("/")[-1]
        path = path[: path.rfind("/") + 1]

    match type:
        case "pkl":
            import pickle

            with open(f"{path}{name}.pkl", "rb") as f:
                print(f"Loaded {name} from {path}")
                return pickle.load(f)
        case "jsonl":
            import jsonlines

            with jsonlines.open(f"{path}{name}.jsonl") as reader:
                print(f"Loaded {name} from {path}")
                return list(reader)
        case "faiss":
            import faiss

            index = faiss.read_index(f"{path}{name}.faiss")
            print(f"Loaded {name} from {path}")
            return index
        case "txt":
            with open(f"{path}{name}.txt", "r") as f:
                print(f"Loaded {name} from {path}")
                return f.read()
        case "npy":
            import numpy as np

            with open(f"{path}{name}.txt", "r") as f:
                print(f"Loaded {name} from {path}")
                return np.load(f)


def mkdir(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")
        return True
    return False

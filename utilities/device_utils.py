import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

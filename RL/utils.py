import torch

select_device = "cuda"

if select_device in ["mps", "cuda", "cpu"]: device = torch.device(select_device)
else: device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
import torch
import sys

from RL.baselineNetwork import Baseline
from RL.CNN import CustomCNN, PretrainedCNN, CustomCNNMultiFrame, RobustCNN
from RL.FF import FF

select_device = "cpu"

if select_device in ["mps", "cuda", "cpu"]: device = torch.device(select_device)
else: device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def get_model(name, *args, **kwargs):
    match name:
        case 'baseline': return Baseline(*args, **kwargs)
        case 'FF': return FF(*args, **kwargs)
        case 'customCNN': return CustomCNN(*args, **kwargs)
        case 'pretrainedCNN': return PretrainedCNN(*args, **kwargs)
        case 'customCNN_MF': return CustomCNNMultiFrame(*args, **kwargs)
        case 'robustCNN': return RobustCNN(*args, **kwargs)
        case _: 
            print(f"The model {name} does not exist")
            sys.exit(1)

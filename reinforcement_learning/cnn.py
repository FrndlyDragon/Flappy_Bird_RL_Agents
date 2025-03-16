import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforcement_learning.base_model import BaseAgents
from reinforcement_learning.utils import device
from game.sprites import Bird
from game.utils import window_width, window_height

class ConvolutionModel(nn.modules):
    def __init__(self):
        super().__init__()

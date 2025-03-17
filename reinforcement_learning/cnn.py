import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import flappy_bird_env # noqa
import gymnasium as gym

from reinforcement_learning.base_model import BaseAgents
from reinforcement_learning.utils import device
from game.sprites import Bird
from game.utils import window_width, window_height

#env = gym.make("FlappyBird-v0")

class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = nn.ReLU(self.fc1(x))
        return self.fc2(x)

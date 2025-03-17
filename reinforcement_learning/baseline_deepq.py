import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from reinforcement_learning.base_model import BaseAgents
from reinforcement_learning.utils import device
from game.sprites import Bird
from game.utils import window_width, window_height

torch.autograd.set_detect_anomaly(True)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc1 = nn.Linear(4, 16)
        self.fc4 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = x.to(device)
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        return self.fc4(x)


class BaselineDeepQAgents(BaseAgents):
    def __init__(self, lr=1e-3):
        super().__init__(n_agents=1, lr=lr)
        self.model = BaselineModel().to(device)

        self.feature_shape = (4,)
        self.features = torch.zeros(*self.feature_shape)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.rewards_mean_ma = 0
        self.gamma = 0.3

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_features(self, game):
        """
        if not game.sprites['pipes'].start: 
            _features = torch.ones(self.n_agents, 1) * game.sprites['ground'].timer
            self.features = torch.hstack([torch.tensor([[bird.pos.y, , window_width - bird.pos.x] for bird in self.birds]), _features]).float()
            return
        """
        feature_lists = []
        for bird in self.birds:
            ver_dists_to_pipes = [abs(pipe.pos.y - bird.pos.y) for pipe in game.sprites['pipes'].bottom_pipes if bird.pos.x < (pipe.pos.x + pipe.size[0])]
            hor_dists_to_pipes = [pipe.pos.x - (bird.pos.x + bird.size[0]) for pipe in game.sprites['pipes'].bottom_pipes if bird.pos.x < pipe.pos.x]
            if len(ver_dists_to_pipes) == 0:  
                ver_dists_to_pipe = window_height - bird.pos.y
            else:
                ver_dists_to_pipe = ver_dists_to_pipes[0]
            if len(hor_dists_to_pipes) == 0: 
                hor_dists_to_pipe = window_width - bird.pos.x
            else:
                hor_dists_to_pipe = hor_dists_to_pipes[0]
            feature_lists.append([bird.pos.y, bird.v.y, ver_dists_to_pipe, hor_dists_to_pipe])
        self.features = torch.tensor(feature_lists).float()
     
    def forward(self):
        if np.random.random() < self.epsilon: return 
        probs = self.model(self.features).cpu()
        dist = Categorical(probs)
        self.output = dist.sample()
        for agent_idx in range(self.n_agents):
            if self.birds[agent_idx].alive:
                self.losses[agent_idx].append(dist.log_prob(self.output[agent_idx]))
        return self.output.numpy()

    def backward(self, score_threshold):
        """
        Backwards pass
        """
        raw_reward = torch.tensor([bird.reward for bird in self.birds], dtype=torch.float)
        self.rewards_mean_ma = self.gamma*raw_reward.mean() + (1-self.gamma)*self.rewards_mean_ma
        self.reward = (raw_reward - self.rewards_mean_ma)
        losses = []
        for reward, agent_loss in zip(self.reward, self.losses):
            losses.append(-reward * (torch.cat(agent_loss).mean().view(1,1)))
        loss = torch.cat(losses).mean()# + self.reward.std()
        loss.backward()
        #for param in self.model.parameters():
        #    print(param.grad)
        self.optimizer.step()
        return loss, raw_reward
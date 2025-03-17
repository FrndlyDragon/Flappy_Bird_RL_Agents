import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from old.base_model import BaseAgents
from old.utils import device
from game.sprites import Bird
from game.utils import window_width, window_height, pipe_y_sep

torch.autograd.set_detect_anomaly(True)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.bn = nn.BatchNorm1d(3)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.to(device)
        #x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)


class BaselineAgents(BaseAgents):
    def __init__(self, n_agents=16, lr=1e-3):
        super().__init__(n_agents=n_agents, lr=lr)
        self.model = BaselineModel().to(device)

        self.feature_shape = (4,)
        self.features = torch.zeros(n_agents, *self.feature_shape)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.gamma = 0.95

    def get_features(self, game):
        """
        if not game.sprites['pipes'].start: 
            _features = torch.ones(self.n_agents, 1) * game.sprites['ground'].timer
            self.features = torch.hstack([torch.tensor([[bird.pos.y, , window_width - bird.pos.x] for bird in self.birds]), _features]).float()
            return
        """
        feature_lists = []
        for bird in self.birds:
            # get features
            next_pipes = [pipe for pipe in game.sprites['pipes'].bottom_pipes if bird.pos.x < (pipe.pos.x + pipe.size[0])]
            pipe_x = next_pipes[0].pos.x if len(next_pipes)>0 else window_width
            pipe_y = (next_pipes[0].pos.y - pipe_y_sep/2) if len(next_pipes)>0 else window_height/2 - pipe_y_sep/2
            bird_y = bird.pos.y
            bird_angle = bird.angle
            # manual normalization
            pipe_x = (pipe_x - (window_width-bird.pos.x)/2) / window_width
            pipe_y = (pipe_y - window_height/2) / window_height
            bird_y = (bird_y - window_height/2) / window_height
            bird_angle = (bird_angle) / 90

            feature_lists.append([bird_y, pipe_y, pipe_x, bird_angle])
        self.features = torch.tensor(feature_lists).float()
     
    def forward(self):
        probs = self.model(self.features).cpu()
        dist = Categorical(probs=probs)
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
        self.reward_mean = self.gamma * self.reward_mean + (1 - self.gamma) * raw_reward.mean()
        self.reward_std = self.gamma * self.reward_std + (1- self.gamma) * raw_reward.std()
        rewards = (raw_reward - self.reward_mean) / (self.reward_std + 1e-9)
        losses = []

        # get good agents
        good_agents = []
        for agent_idx in range(self.n_agents):
            if rewards[agent_idx] > 0:
                good_agents.append(agent_idx)

        if len(good_agents) == 0:
            good_agents = torch.argsort(rewards, descending=True)[:max(1, self.n_agents // 4)].tolist()

        for agent_idx in range(self.n_agents): # good_agents:
            log_prob = torch.cat(self.losses[agent_idx]).mean().view(1, 1)
            loss = -log_prob * rewards[agent_idx]
            losses.append(loss)

        loss = torch.cat(losses).mean()# + self.reward.std()
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    print(param.grad)
        for agent_idx in range(self.n_agents):
            self.losses[agent_idx] = []
        self.optimizer.step()
        return loss, raw_reward
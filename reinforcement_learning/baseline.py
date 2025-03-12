import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforcement_learning.base_model import BaseAgents
from reinforcement_learning.utils import device
from game.sprites import Bird
from game.utils import window_width, window_height

torch.autograd.set_detect_anomaly(True)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
    
    def forward(self, x):
        x = x.to(device)
        return F.softmax(self.fc1(x), dim=-1)


class BaselineAgents(BaseAgents):
    def __init__(self, n_agents=16, lr=1e-3):
        super().__init__(n_agents=n_agents, lr=lr)
        self.model = BaselineModel().to(device)

        self.feature_shape = (3,)
        self.features = torch.zeros(n_agents, *self.feature_shape)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.rewards_mean_ma = 0
        self.gamma = 0.5

    def get_features(self, game):
        if not game.sprites['pipes'].start: 
            _features = torch.ones(self.n_agents, 2) * torch.tensor([window_width, game.sprites['ground'].timer], dtype=torch.float)
            self.features = torch.hstack([torch.tensor([[window_height - bird.pos.y] for bird in self.birds]), _features]).float()
            return
        feature_lists = [[[abs(pipe.pos.y - bird.pos.y) for pipe in game.sprites['pipes'].get_all_pipes() if bird.pos.x < pipe.pos.x + pipe.size[0]][0], [pipe.pos.x - (bird.pos.x + bird.size[0]) for pipe in game.sprites['pipes'].get_all_pipes() if bird.pos.x < pipe.pos.x][0]] for bird in self.birds]
        self.features = torch.hstack([torch.tensor(feature_lists), torch.ones(self.n_agents, 1) * game.sprites['ground'].timer]).float()
     
    def forward(self):
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
        self.reward = torch.tensor([bird.score - score_threshold for bird in self.birds], dtype=torch.float)
        #self.rewards_mean_ma = self.gamma*raw_reward.mean() + (1-self.gamma)*self.rewards_mean_ma
        #self.reward = (raw_reward - raw_reward.mean())/(raw_reward.std() + 1e-8)
        losses = []
        for reward, agent_loss in zip(self.reward, self.losses):
            losses.append(-reward * (torch.cat(agent_loss).mean().view(1,1) + 1))
        loss = torch.cat(losses).mean()
        loss.backward()
        #for param in self.model.parameters():
        #    print(param.grad)
        self.optimizer.step()
        return loss, self.reward
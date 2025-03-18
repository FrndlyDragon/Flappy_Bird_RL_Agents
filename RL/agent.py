from RL.policyNetwork import *
from RL.utils import device
import torch.optim as optim
import torch
import numpy as np

class REINFORCE: 
    def __init__(self, network='baseline', lr=0.01, gamma=0.99, epsilon_exploration = False, epsilon_start=1.0, epsilon_end =0.001, epsilon_decay = 0.9) -> None:
        match network:
            case 'baseline': self.policy = Baseline().to(device)
            case 'CNN': self.policy = CNN().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.epsilon_exploration = epsilon_exploration
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.log_probs = []
        self.rewards = []
    
    def input_type(self):
        return self.policy.input_type()
    
    def select_action(self, state, training= True):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        if self.epsilon_exploration and training and np.random.random() < self.epsilon:
            action = np.random.choice(range(action_probs.shape[0]))
            log_prob = action_dist.log_prob(torch.tensor(action))
        else:
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action))
        self.log_probs.append(log_prob)
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        R = 0    
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)   
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob*R)
        loss = torch.stack(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
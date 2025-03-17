import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self,) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        X = torch.relu(self.fc1(state))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs


class REINFORCE: 
    def __init__(self, lr=0.01, gamma=0.99) -> None:
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action))  # Store log probability
        return action.item()
    
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
        
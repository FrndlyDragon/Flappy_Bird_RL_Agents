from RL.policyNetwork import *
import torch.optim as optim
import torch

class REINFORCE: 
    def __init__(self, network='baseline', lr=0.01, gamma=0.99) -> None:
        match network:
            case 'baseline': self.policy = Baseline()
            case 'CNN': self.policy = CNN()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
    
    def input_type(self):
        return self.policy.input_type()
    
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
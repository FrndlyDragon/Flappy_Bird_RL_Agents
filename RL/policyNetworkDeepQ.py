import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self,) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, state):
        X = torch.relu(self.fc1(state))
        X = torch.relu(self.fc2(X))
        action_probs = self.fc3(X)
        return action_probs


class REINFORCE_DEEPQ: 
    def __init__(self, lr=0.01, gamma=0.99) -> None:
        self.policy = PolicyNetwork()
        self.target = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.target_update_freq = 250
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([random.randint(0, 1)])
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy(state)
            return torch.argmax(q_values).item() 
    
    def store_reward(self, reward):
        self.rewards.append(reward)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def update_policy(self, state, action, reward, next_state, terminated, iterations):
        self.memory.append((state, action, reward, next_state, terminated))

        if len(self.memory) < self.batch_size: return 

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        terminated_batch = torch.FloatTensor(terminated_batch)

        q_values = self.policy(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            max_next_q_values = self.target(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - terminated_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if iterations % self.target_update_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())
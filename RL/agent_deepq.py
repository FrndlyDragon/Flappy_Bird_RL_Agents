import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
from collections import deque

from RL.policyNetwork import *
from RL.utils import device

class REINFORCE_DEEPQ: 
    def __init__(self, network='baseline', lr=0.01, gamma=0.99, epsilon_exploration=False, epsilon_start=1.0, epsilon_end =0.001, epsilon_decay = 0.9) -> None:
        self.network = network
        match network:
            case 'baseline': 
                self.policy = Baseline(deepq=True).to(device)
                self.target = Baseline(deepq=True).to(device)
            case 'CNN': 
                self.policy = CNN(deepq=True).to(device)
                self.target = CNN(deepq=True).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = 250
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return torch.tensor(random.randint(0, 1))
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy(state)
            return torch.argmax(q_values).item() 
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def input_type(self):
        return self.policy.input_type()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def update_policy(self, state, action, reward, next_state, terminated, iterations):
        self.memory.append((state, action, reward, next_state, terminated))

        if len(self.memory) < self.batch_size: return 

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
        terminated_batch = torch.FloatTensor(np.array(terminated_batch)).to(device)

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
import torch
import torch.nn as nn
import numpy as np 

import pygame

class FF(nn.Module):
    def __init__(self, deepq=False) -> None:
        super(FF, self).__init__()
        self.shape = (72,100)
        self.repr_dim = 300

        self.fc1 = nn.Linear(self.shape[0]*self.shape[1], 300)
        self.dropout = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, self.repr_dim)
        self.fc4 = nn.Linear(self.repr_dim, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)
    
    def pretrain_forward(self, state):
        X = state.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.dropout(X)
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        return X

    def forward(self, state):
        X = state.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.dropout(X)
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        action_probs = self.softmax(self.fc4(X))
        return action_probs
    
    def input_type(self):
        return "img"
    
    def get_input(self, game):
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        gray_pixels = np.expand_dims(np.mean(pixels, axis=2), axis=0)
        return gray_pixels
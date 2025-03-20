import torch
import torch.nn as nn
import numpy as np 

import pygame

class FF(nn.Module):
    def __init__(self, deepq=False) -> None:
        super(FF, self).__init__()
        self.nframes = 1 # multiple frames handling not implemented
        self.shape = (80,80)
        self.repr_dim = 6

        # pretrain
        self.p_fc1 = nn.Linear(self.shape[0]*self.shape[1], 512)
        self.p_fc2 = nn.Linear(512, 256)
        self.p_fc3 = nn.Linear(256, 128)
        self.p_fc4 = nn.Linear(128, self.repr_dim)
        # train
        self.fc1 = nn.Linear(self.repr_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)

        self.pretrain = [self.p_fc1, self.p_fc2, self.p_fc3, self.p_fc4]
    
    def pretrain_forward(self, state):
        X = state.view(state.shape[0], -1)
        X = torch.relu(self.p_fc1(X))
        X = torch.relu(self.p_fc2(X))
        X = torch.relu(self.p_fc3(X))
        X = torch.tanh(self.p_fc4(X))
        return X

    def forward(self, state):
        X = state.view(state.shape[0], -1)
        X = torch.relu(self.p_fc1(X))
        X = torch.relu(self.p_fc2(X))
        X = torch.relu(self.p_fc3(X))
        X = torch.tanh(self.p_fc4(X))
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def input_type(self):
        return "img"
    
    def get_input(self, game):
        # get frame
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        gray_pixels = np.expand_dims(np.mean(pixels, axis=2), axis=0)
        # normalize
        frame = gray_pixels / 255.0
        frame = (frame - np.mean(frame)) / (np.std(frame) + 1e-9)
        return frame
    
    def freeze_pretrain(self):
        for layer in self.pretrain:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_pretrain(self):
        for layer in self.pretrain:
            for param in layer.parameters():
                param.requires_grad = True
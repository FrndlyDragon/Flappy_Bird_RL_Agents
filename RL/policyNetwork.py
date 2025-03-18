import torch
import torch.nn as nn

import numpy as np

import pygame
from game.utils import window_width, window_height

class Baseline(nn.Module):
    def __init__(self, deepq=False) -> None:
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        X = torch.relu(self.fc1(state))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def input_type(self):
        return "var"
    
    def get_input(self, game):
        next_top_pipes = [pipe for pipe in game.pipes.top_pipes if game.bird.pos.x < (pipe.pos.x + pipe.size[0])]
        next_bottom_pipes = [pipe for pipe in game.pipes.bottom_pipes if game.bird.pos.x < (pipe.pos.x + pipe.size[0])]
        ### get features
        # top pipe
        top_pipe_x = next_top_pipes[0].pos.x if len(next_top_pipes)>0 else window_width - game.bird.pos.x
        top_pipe_y = next_top_pipes[0].pos.y + 611 if len(next_top_pipes)>0 else window_height/2
        # bottom pipe
        bottom_pipe_x = next_bottom_pipes[0].pos.x if len(next_bottom_pipes)>0 else window_width - game.bird.pos.x
        bottom_pipe_y = next_bottom_pipes[0].pos.y if len(next_bottom_pipes)>0 else window_height/2
        # bird
        bird_y = game.bird.pos.y
        bird_angle = game.bird.angle
        ### manual normalization
        # top pipe
        top_pipe_x = (top_pipe_x - (window_width - game.bird.pos.x)/2) / window_width
        top_pipe_y = (top_pipe_y - window_height/2) / window_height
        # bottom pipe
        bottom_pipe_x = (bottom_pipe_x - (window_width - game.bird.pos.x)/2) / window_width
        bottom_pipe_y = (bottom_pipe_y - window_height/2) / window_height
        bird_y = (bird_y - window_height/2) / window_height
        bird_angle = (bird_angle) / 90

        return [bird_y, bird_angle, top_pipe_x, top_pipe_y, bottom_pipe_x, bottom_pipe_y]
    

class CNN(nn.Module):
    def __init__(self, deepq=False) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = torch.relu(self.conv3(X))
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def input_type(self):
        return "img"
    
    def get_input(self, game, shape=(84,84)):
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), (shape[0], shape[1]))
        pixels = pygame.surfarray.array3d(pixels)
        gray_pixels = np.expand_dims(np.mean(pixels, axis=2), axis=0)
        return gray_pixels
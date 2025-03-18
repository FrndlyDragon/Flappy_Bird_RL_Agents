import torch
import torch.nn as nn
import numpy as np
import pygame
from torchvision import models


class CNN(nn.Module):

    def __init__(self) -> None:
        super(CNN, self).__init__()

    def _initialize_weights(self):
        """
        Initialize weights using He initialization for better training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state) -> None:
        pass

    def input_type(self) -> str:
        return 'img'
    
    def get_input(self, game, shape=(64,64)):
        # get image
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), (shape[0], shape[1]))
        pixels = pygame.surfarray.array3d(pixels)
        current_frame = np.expand_dims(np.mean(pixels, axis=2), axis=0)
        # normalize image
        current_frame /= 255
        current_frame = (current_frame - np.mean(current_frame)) / (np.std(current_frame) + 1e-8)

        if self.previous_frame is None:
            self.previous_frame = current_frame
        stacked_frames = np.vstack((self.previous_frame, current_frame))
        self.previous_frame = current_frame
        return stacked_frames


class CustomCNN(CNN):
    def __init__(self, deepq=False, hidden_size=128, dropout_rate=0.2) -> None:
        super(CustomCNN, self).__init__()
        self.previous_frame = None

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=8, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) 
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(1152, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 2, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

    def forward(self, state):
        X = torch.relu(self.bn1(self.conv1(state)))
        X = torch.relu(self.bn2(self.conv2(X)))
        X = torch.relu(self.bn3(self.conv3(X)))
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.dropout1(X)
        X = torch.relu(self.fc2(X))
        X = self.dropout2(X)
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    

class PretrainedCNN(CNN):
    def __init__(self, deepq=False, hidden_size=512, dropout_rate=0.2) -> None:
        super(PretrainedCNN, self).__init__()
        self.previous_frame = None

        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base_model.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.fc1 = nn.Linear(5120, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 4, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

    def forward(self, state):
        X = self.feature_extractor(state)
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.dropout1(X)
        X = torch.relu(self.fc2(X))
        X = self.dropout2(X)
        action_probs = self.softmax(self.fc3(X))
        return action_probs
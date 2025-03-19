import torch
import torch.nn as nn
import numpy as np
import pygame
from torchvision import models

class CNN(nn.Module):

    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.shape = (84,84)
        self.pretrain = []

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
    
    def freeze_pretrain(self):
        for layer in self.pretrain:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_pretrain(self):
        for layer in self.pretrain:
            for param in layer.parameters():
                param.requires_grad = True

class CustomCNN(CNN):
    def __init__(self, deepq=False) -> None:
        super(CustomCNN, self).__init__()
        self.repr_dim = 6 
        self.shape = (80,80)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2) 
        self.fc1 = nn.Linear(9248, 6)
        self.fc2 = nn.Linear(6, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

        self.pretrain = [self.conv1, self.conv2, self.fc1]

    def pretrain_forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = X.view(state.shape[0], -1)
        X = self.fc1(X)
        return X

    def forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        action_probs = self.softmax(X)
        return action_probs
    
    def get_input(self, game):
        # get image
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        current_frame = np.expand_dims(np.mean(pixels, axis=2), 0)
        # normalize image
        current_frame /= 255
        current_frame = (current_frame - np.mean(current_frame)) / (np.std(current_frame) + 1e-8)
        return current_frame
    
class CustomCNNMultiFrame(CNN):
    def __init__(self, deepq=False, nframes=2) -> None:
        super(CustomCNNMultiFrame, self).__init__()
        self.nframes = nframes
        self.previous_frames = [np.zeros(self.shape) for _ in range(nframes)]
        self.repr_dim = 6 
        self.shape = (80,80)

        self.conv1 = nn.Conv2d(in_channels=nframes, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2) 
        self.fc1 = nn.Linear(9248, 6)
        self.fc2 = nn.Linear(6, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)

        #self.pretrain = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.fc1, self.fc2]
        self._initialize_weights()

        self.pretrain = [self.conv1, self.conv2, self.fc1]

    def pretrain_forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = X.view(state.shape[0], -1)
        X = self.fc1(X)
        return X

    def forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        action_probs = self.softmax(X)
        return action_probs
    
    def get_input(self, game):
        # get image
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        current_frame = np.mean(pixels, axis=2)
        # normalize image
        current_frame /= 255
        current_frame = (current_frame - np.mean(current_frame)) / (np.std(current_frame) + 1e-8)

        self.previous_frames.pop(0)
        self.previous_frames.append(current_frame)
        return self.previous_frames
    
    def reset_memory(self):
        self.previous_frames = [np.zeros(self.shape) for _ in range(self.nframes)]

class PretrainedCNN(CNN):
    def __init__(self, deepq=False, hidden_size=512, dropout_rate=0.2) -> None:
        super(PretrainedCNN, self).__init__()
        self.repr_dim = hidden_size 

        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.fc1 = nn.Linear(5120, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 4, 2)
        if deepq: self.softmax = lambda x:x
        else: self.softmax = nn.Softmax(dim=-1)
        self._initialize_weights()

        self.pretrain = [self.feature_extractor, self.fc1]

    def pretrain_forward(self, state):
        X = self.feature_extractor(state)
        X = X.view(state.shape[0], -1)
        X = self.fc1(X)
        return X

    def forward(self, state):
        X = self.feature_extractor(state)
        X = X.view(state.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.dropout1(X)
        X = torch.relu(self.fc2(X))
        X = self.dropout2(X)
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def get_input(self, game):
        # get image
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        current_frame = np.mean(pixels, axis=2)
        # normalize image
        current_frame /= 255
        current_frame = (current_frame - np.mean(current_frame)) / (np.std(current_frame) + 1e-8)
        return current_frame

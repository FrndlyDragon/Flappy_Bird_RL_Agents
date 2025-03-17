import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self,) -> None:
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        X = torch.relu(self.fc1(state))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def input_type(self):
        return "var"
    

class CNN(nn.Module):
    def __init__(self,) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        X = torch.relu(self.conv1(state))
        X = torch.relu(self.conv2(X))
        X = torch.relu(self.conv3(X))
        X = X.view(1, -1)
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        action_probs = self.softmax(self.fc3(X))
        return action_probs
    
    def input_type(self):
        return "img"
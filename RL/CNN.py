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
        self.nframes = 1
        self.repr_dim = 6
        self.shape = (80,80)

        #conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.feature_fc1 = nn.Linear(2304, 512)
        self.feature_fc2 = nn.Linear(512, self.repr_dim)
        
        self.policy_fc1 = nn.Linear(self.repr_dim, 128)
        self.policy_fc2 = nn.Linear(128, 64)
        self.policy_fc3 = nn.Linear(64, 2)

        if deepq: 
            self.softmax = lambda x: x  # for DeepQ, no softmax (using Q-values directly)
        else: 
            self.softmax = nn.Softmax(dim=-1)  # For policy gradient
            
        self._initialize_weights()
        
        self.pretrain = [self.conv1, self.conv2, self.conv3, self.feature_fc1, self.feature_fc2]

    def pretrain_forward(self, state):
        """Forward pass for pretraining - extract features only"""
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(state.shape[0], -1)
        x = torch.relu(self.feature_fc1(x))
        features = torch.relu(self.feature_fc2(x))
        return features

    def forward(self, state):
        """Full forward pass for action selection"""
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(state.shape[0], -1)
        x = torch.relu(self.feature_fc1(x))
        features = torch.relu(self.feature_fc2(x))
        
        x = torch.relu(self.policy_fc1(features))
        x = self.policy_fc2(x)
        x = self.policy_fc3(x)
        
        action_probs = self.softmax(x)
        return action_probs
    
    def get_input(self, game):
        """Process the game screen into network input"""
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        
        # Convert to grayscale
        current_frame = np.expand_dims(np.mean(pixels, axis=2), 0)
        
        # Normalize image (important for CNN performance)
        current_frame = current_frame / 255.0
        current_frame = (current_frame - np.mean(current_frame)) / np.std(current_frame + 1e-9)
        return current_frame

class CustomCNNMultiFrame(CNN):
    def __init__(self, deepq=False, nframes=4) -> None:
        super(CustomCNNMultiFrame, self).__init__()
        self.nframes = nframes
        self.previous_frames = [np.zeros(self.shape) for _ in range(nframes)]
        self.repr_dim = 5
        self.shape = (80,80)

        self.conv1 = nn.Conv2d(in_channels=nframes, out_channels=32, kernel_size=8, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

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
        X = self.maxpool1(torch.relu(self.conv1(state)))
        X = self.maxpool2(torch.relu(self.conv2(X)))
        X = self.maxpool3(torch.relu(self.conv3(X)))
        X = X.view(state.shape[0], -1)
        X = self.fc1(X)
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
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
    
class RobustCNN(CNN):
    def __init__(self, deepq=False) -> None:
        super(RobustCNN, self).__init__()
        self.shape = (84, 84)  # Taille standard pour les jeux
        
        # Architecture inspirée du DQN pour Atari
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # Normalisation par batch pour stabilité
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Couche adaptive pour s'adapter à différentes tailles d'entrée
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Calcul automatique de la taille après convolutions
        conv_output_size = 64 * 7 * 7  # après l'adaptive pooling
        
        # Couches fully connected
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(0.2)  # Dropout pour éviter le surapprentissage
        
        self.fc2 = nn.Linear(512, 2)  # 2 actions: flap ou no flap
        
        # Fonction d'activation
        self.activation = nn.LeakyReLU(0.1)  # LeakyReLU pour éviter les neurones morts
        
        # Couche de sortie
        if deepq:
            self.softmax = lambda x: x  # Pas de softmax pour DeepQ
        else:
            self.softmax = nn.Softmax(dim=-1)
            
        self._initialize_weights()
        
        # Pas de couches pretrain car ce modèle n'en a pas besoin
        self.pretrain = []

    def forward(self, state):
        """Full forward pass pour l'inférence"""
        # Extraction de caractéristiques
        x = self.activation(self.bn1(self.conv1(state)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        
        # Pooling adaptatif pour garantir une taille de sortie constante
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Aplatissement
        
        # Couches fully connected
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        # Appliquer softmax si nécessaire (pour policy gradient)
        action_probs = self.softmax(x)
        return action_probs
    
    def pretrain_forward(self, state):
        """Pour compatibilité avec le code existant, même si pas utilisé"""
        x = self.activation(self.bn1(self.conv1(state)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        return x
    
    def get_input(self, game):
        """Prétraitement de l'image du jeu"""
        # Récupérer l'image
        pixels = pygame.surfarray.array3d(game.screen)
        pixels = pygame.transform.scale(pygame.surfarray.make_surface(pixels), self.shape)
        pixels = pygame.surfarray.array3d(pixels)
        
        # Conversion en niveaux de gris
        current_frame = np.expand_dims(np.mean(pixels, axis=2), 0)
        
        # Normalisation importante pour les CNNs
        current_frame = current_frame / 255.0
        
        # Normalisation avancée (zero-center + division par écart-type)
        if np.std(current_frame) > 0:
            current_frame = (current_frame - np.mean(current_frame)) / np.std(current_frame)
        
        return current_frame
        
    def reset(self):
        """Réinitialisation si nécessaire"""
        pass
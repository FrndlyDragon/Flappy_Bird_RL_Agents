import numpy as np


class DeepQLearning:

    def __init__(self, lr= 5e-2, gamma= 0.95, epsilon= 0.1) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon



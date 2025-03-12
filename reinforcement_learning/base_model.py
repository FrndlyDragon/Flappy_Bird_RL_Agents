import torch
import numpy as np

from game.sprites import BaseSprite, Bird

class BaseAgents(BaseSprite):
    def __init__(self, n_agents=16, lr=1e-3):
        self.n_agents = n_agents
        self.bird_alpha = 1/np.sqrt(n_agents)
        self.birds = np.array([Bird(alpha=self.bird_alpha) for _ in range(n_agents)])
        self.alive_filter = np.ones(self.n_agents)
        self.model = None

        self.feature_shape = (0,)
        self.features = torch.zeros(n_agents, *self.feature_shape)
        self.output = None 
        self.losses = [[] for _ in range(n_agents)]
        self.reward = torch.zeros(n_agents)

        self.loss_fn = None
        self.optimizer = None
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.losses = [[] for _ in range(self.n_agents)]
    
    def get_features(self, game):
        """
        Called every frame, compute features
        """
        pass
    
    def forward(self):
        """
        Should be set as FlappyBird.get_event, computes the forward pass + aggregates gradients
        """

    def backwards(self, score_threshold):
        """
        Backwards pass
        """
        pass

    def update(self, dt, keydown, sprites):
        all_dead = True
        for bird, kd in zip(self.birds, keydown):
            all_dead &= bird.update(dt, kd, sprites)
        return all_dead
    
    def blit(self, screen, debug_kwargs):
        for bird in self.birds:
            screen.blit(bird.image, bird.pos.as_tuple())
            if debug_kwargs['hitbox_show']: bird.hitbox.blit(screen)
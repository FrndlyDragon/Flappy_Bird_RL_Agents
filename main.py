import pygame
import numpy as np

import time

from game.fb_game import FlappyBird
from reinforcement_learning.baseline import BaselineAgents

if __name__ == "__main__":           
    pygame.init() 
    pygame.display.set_caption("Flappy Bird") 

    game = FlappyBird(debug_kwargs={'hitbox_show': False}, max_speed=True, max_frames=3200)
    agents = BaselineAgents(n_agents=64, lr=5e-2)
    game.set_rl(agents.forward, agents)
    
    epochs = 500

    for epoch in range(epochs):
        agents.zero_grad()
        game.reset()
        #time.sleep(0.1)
        game.run()
        loss, reward = agents.backward(100)
        _loss_str = f"{loss:.2f}"
        _reward_str = f"{max(reward):.2f}"
        _time_str = f"{max([bird.timer for bird in agents.birds]):.2f}"
        print(f"Epoch {epoch+1:>3}/{epochs}, Loss: {_loss_str:>8}, Best Performing Agent Score/Time: {_reward_str:>6}/{_time_str:>5}")
import pygame
from game.fb_game import FlappyBird
from game.dynamicRules import DynamicRules
from RL.policyNetwork import *



if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Flappy Bird") 
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, max_speed=True)
    dynamicRules = DynamicRules(pipe_y_sep=350, score_threshold=5, upd_value=10)

    agent = REINFORCE(lr=1e-2)

    epochs = 5000

    score_mean = 0
    for epoch in range(epochs):
        changed = dynamicRules.update(score_mean)
        if changed: score_mean = 0

        state = game.reset()
        total_reward = 0
 
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, kwargs = game.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            

            if terminated or kwargs['score']>100:
                agent.update_policy()
                break
        score_mean = 0.1*kwargs['score'] + 0.9*score_mean
        print(f"Epoch {epoch}, Total reward {total_reward:4f}, Score moving average {score_mean:4f}")
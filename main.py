import pygame
from game.fb_game import FlappyBird
from game.dynamicRules import DynamicRules
from RL.agent import REINFORCE
from RL.agent_deepq import REINFORCE_DEEPQ

mode = "deepq"  # "policy_grad" or "deepq"

"""
Hyperparams (DeepQ):

baseline: 
- batch_size: 64
- target_update_freq: 250
"""

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Flappy Bird") 
    if mode == "deepq": model = REINFORCE_DEEPQ
    elif mode == "policy_grad": model = REINFORCE
    else: raise ValueError(f'{mode} not implemented, only "policy_grad" or "deepq"')
    agent = model(network='baseline', lr=3e-3)
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=True)
    dynamicRules = DynamicRules(pipe_y_sep=250, score_threshold=5, upd_value=25)    

    epochs = 5000

    score_mean = 0
    for epoch in range(epochs):
        changed = dynamicRules.update(score_mean)
        if changed: score_mean = 0

        state = game.reset()
        total_reward = 0
    
        iteration_count = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pass

            action = agent.select_action(state)
            next_state, reward, terminated, kwargs = game.step(action)
            agent.store_reward(reward)
            if mode == "deepq": agent.update_policy(state, action, reward, next_state, terminated, iteration_count)
            state = next_state
            total_reward += reward

            if terminated or kwargs['score']>100:
                if mode == "policy_grad" : agent.update_policy()
                break
            
            iteration_count += 1
        if mode == "deepq": agent.decay_epsilon()
        score_mean = 0.1*kwargs['score'] + 0.9*score_mean
        _reward_str = f"{total_reward:4f}"
        _score_str = f"{score_mean:4f}"
        print(f"Epoch {epoch:>4}, Total reward {_reward_str:>9}, Score moving average {_score_str:>9}")
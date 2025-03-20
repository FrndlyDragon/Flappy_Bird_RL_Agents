import pygame
import copy
from tqdm import tqdm

from game.fb_game import FlappyBird
from game.dynamicRules import DynamicRules

def train(agent, epochs=1000, score_gamma =0.95, max_speed=True):
    agent.policy.train()
    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=max_speed)
    dynamicRules = DynamicRules() 

    scores = []
    rule_change_epochs = []
    score_mean = 0
    best_score_mean = 0
    best_policy = copy.deepcopy(agent.policy)
    for epoch in range(epochs):
        changed_rules = dynamicRules.update(score_mean)
        if changed_rules:
            rule_change_epochs.append(epoch)
            score_mean = 0
            # reset best policy due to change of environment
            best_policy = copy.deepcopy(agent.policy)
            best_score_mean = 0
        state = game.reset()

        total_reward = 0
        iteration_count = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, kwargs = game.step(action)
            if agent.mode == "deepq": agent.update_policy(state, action, reward, next_state, terminated, iteration_count)
            state = next_state
            agent.store_reward(reward)
            total_reward += reward
            if terminated or kwargs['score']>50:
                if agent.mode == "policy_grad": agent.update_policy(state, action, reward, next_state, terminated, epoch)
                break
            iteration_count += 1
        score_mean = (1-score_gamma)*kwargs['score'] + score_gamma*score_mean
        scores.append(kwargs['score'])
        if score_mean > best_score_mean:
            best_policy = copy.deepcopy(agent.policy)
            best_score_mean = score_mean

        _rw_str = f"{total_reward:4f}"
        _sc_str = f"{score_mean:4f}"
        print(f"Epoch {epoch:>4}, Total reward {_rw_str:>10}, Score moving average {_sc_str:>10}")
        
    print(f"Finished training, best mean score {best_score_mean}")
    return best_policy, scores, rule_change_epochs


def eval(agent, policy, n_games = 20, max_score=1000, max_speed=True):
    agent.policy.load_state_dict(copy.deepcopy(policy.state_dict()))
    agent.policy.eval()

    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    DynamicRules().default_rules()
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=max_speed)

    total_score = 0
    for _ in tqdm(range(n_games)):
        state = game.reset()

        while True:
            action = agent.select_action(state, training=False)
            state, _, terminated, kwargs = game.step(action)
            if terminated or kwargs['score']>max_score:
                break
        total_score += kwargs['score']
    
    print(f'Average score for {n_games} games : {total_score/n_games:3f}')

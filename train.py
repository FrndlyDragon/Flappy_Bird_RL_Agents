import pygame
import RL.agent as agent
from game.fb_game import FlappyBird
from game.dynamicRules import DynamicRules
import copy


def train(agent, epochs=1000, score_gamma =0.9):
    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=True)
    dynamicRules = DynamicRules(pipe_y_sep=275, score_threshold=5, upd_value=25)    

    scores = []
    rule_change_epochs = []
    score_mean = 0
    best_score_mean = 0
    best_policy = agent.policy
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
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, kwargs = game.step(action)
            state = next_state
            agent.store_reward(reward)
            total_reward += reward
            if terminated or kwargs['score']>100:
                agent.update_policy(state, action, reward, next_state, terminated, epoch)
                break

        score_mean = (1-score_gamma)*kwargs['score'] + score_gamma*score_mean
        scores.append(kwargs['score'])
        if score_mean > best_score_mean:
            best_policy = copy.deepcopy(agent.policy)
            best_score_mean = score_mean

        _rw_str = f"{total_reward:4f}"
        _sc_str = f"{score_mean:4f}"
        print(f"Epoch {epoch}, Total reward {_rw_str:>10}, Score moving average {_sc_str:>10}")
        
    print(f"Finished training, best mean score {best_score_mean}")
    return best_policy, scores, rule_change_epochs


def eval(agent, policy, n_games = 20, max_score=1000):
    agent.policy = policy

    DynamicRules().default_rules()
    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    game = FlappyBird(debug_kwargs={'hitbox_show': False}, agent=agent, state_type=agent.input_type(), max_speed=True)

    total_score = 0
    for _ in range(n_games):
        state = game.reset()

        while True:
            action = agent.select_action(state, training=False)
            state, _, terminated, kwargs = game.step(action)
            if terminated or kwargs['score']>max_score:
                break
        total_score += kwargs['score']
    
    print(f'Average score for {n_games} games : {total_score/n_games:3f}')

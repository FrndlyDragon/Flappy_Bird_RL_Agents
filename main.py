import torch
from RL.agent import REINFORCE
from train import train, eval
from pretrain import pretrain
from util import *
from RL.agent_deepq import REINFORCE_DEEPQ
from game.dynamicRules import DynamicRules
import pygame

"""
Hyperparams (DeepQ):

baseline: 
- lr: 1e-3
- epochs: 200
- batch_size: 64
- target_update_freq: 250
- epsilon_decay: 0.9
"""

<<<<<<< HEAD
network = 'customCNN'  # "baseline" or "customCNN" or "pretrainedCNN" or FF
=======
network = 'customCNN_MF'  # "baseline" or "customCNN" or "customCNN_MF" or "pretrainedCNN" or FF
>>>>>>> cab3a20793287758d88cfd6b824ed083139d0cbe
mode = "deepq"  # "policy_grad" or "deepq"

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Flappy Bird")

    if mode == "deepq": model = REINFORCE_DEEPQ
    elif mode == "policy_grad": model = REINFORCE
    else: raise ValueError(f'{mode} not implemented, only "policy_grad" or "deepq"')

<<<<<<< HEAD
    agent = model(network=network, lr=5e-3, batch_size=128, target_update_freq=150, epsilon_decay=0.99, epsilon_exploration=False)

    pretrain(agent, epochs=30, dataset_size=5000, batch_size=64, lr=2e-3,
             save_dataset=True, use_saved=True, dataset_path="data/pretrained_dataset_signle_frame_84.pth", nframes=1)
    #agent.policy.freeze_pretrain()
=======
    agent = model(network=network, lr=5e-3, batch_size=64, target_update_freq=200, epsilon_decay=0.95, epsilon_exploration=True)
    DynamicRules().set_rules(275, 5, 25)

    if network != 'baseline':
        pretrain(agent, epochs=50, dataset_size=5000, batch_size=64, lr=2e-3,
                save_dataset=True, use_saved=False, dataset_path="pretrained_dataset_4_frames.pth", nframes=4)
>>>>>>> cab3a20793287758d88cfd6b824ed083139d0cbe

    # train
    policy, mean_scores, rulechange_epochs = train(agent, 2000)

    # plot
    plot_performance(mean_scores, points=rulechange_epochs, fname=f'{mode}_{network}_scores')

    # eval
    eval(agent, policy, n_games=100)

    # save model
    torch.save(policy.state_dict(), f"./models/{mode}_{network}.pth")

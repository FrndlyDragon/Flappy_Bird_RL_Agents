import torch
from RL.agent import REINFORCE
from train import train, eval
from pretrain import pretrain
from util import *
from RL.agent_deepq import REINFORCE_DEEPQ
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

network = 'customCNN_MF'  # "baseline" or "customCNN" or "pretrainedCNN" or FF
mode = "deepq"  # "policy_grad" or "deepq"

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Flappy Bird")

    if mode == "deepq": model = REINFORCE_DEEPQ
    elif mode == "policy_grad": model = REINFORCE
    else: raise ValueError(f'{mode} not implemented, only "policy_grad" or "deepq"')

    agent = model(network=network, lr=1e-3, batch_size=64, target_update_freq=250, epsilon_decay=0.995, epsilon_exploration=False)

    pretrain(agent, epochs=50, dataset_size=5000, batch_size=64, lr=2e-4,
             save_dataset=True, use_saved=True, dataset_path="data/pretrained_dataset_3_frames.pth", nframes=3)
    agent.policy.freeze_pretrain()

    # train
    policy, mean_scores, rulechange_epochs = train(agent, 1000)

    # plot
    plot_performance(mean_scores, points= rulechange_epochs, fname=f'{mode}_{network}_scores')

    # eval
    eval(agent, policy, n_games=100)

    # save model
    torch.save(policy.state_dict(), f"./models/{mode}_{network}.pth")

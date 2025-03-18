import torch
from RL.agent import REINFORCE
from train import *
from util import *
from RL.agent_deepq import REINFORCE_DEEPQ

"""
Hyperparams (DeepQ):

baseline: 
- lr: 3e-3
- batch_size: 64
- target_update_freq: 250
"""

network = 'CNN'  # "baseline" or "CNN"
mode = "deepq"  # "policy_grad" or "deepq"

if __name__ == "__main__":

    if mode == "deepq": model = REINFORCE_DEEPQ
    elif mode == "policy_grad": model = REINFORCE
    else: raise ValueError(f'{mode} not implemented, only "policy_grad" or "deepq"')

    agent = model(network=network, lr=1e-2, batch_size=128, target_update_freq=100)

    # train
    policy, mean_scores = train(agent, 1500)

    # plot
    plot_performance(mean_scores, f'{network}_mean_scores')

    # eval
    eval(agent, policy, n_games=100)

    # save model
    torch.save(policy.state_dict(), f"./models/{network}.pth")

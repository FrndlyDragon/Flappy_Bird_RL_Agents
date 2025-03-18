import torch
from RL.agent import REINFORCE
from train import *
from util import *
from RL.agent_deepq import REINFORCE_DEEPQ

"""
Hyperparams (DeepQ):

baseline: 
- batch_size: 64
- target_update_freq: 250
"""

network = 'baseline'
mode = "policy_grad"  # "policy_grad" or "deepq"

if __name__ == "__main__":

    if mode == "deepq": model = REINFORCE_DEEPQ
    elif mode == "policy_grad": model = REINFORCE
    else: raise ValueError(f'{mode} not implemented, only "policy_grad" or "deepq"')

    agent = model(network=network, lr=5e-3, epsilon_exploration=False)

    # train
    policy, mean_scores, rulechange_epochs = train(agent, 1000)

    # plot
    plot_performance(mean_scores, points= rulechange_epochs, fname=f'{network}_scores')

    # eval
    eval(agent, policy, n_games=100)

    # save model
    torch.save(policy.state_dict(), f"./models/{network}.pth")

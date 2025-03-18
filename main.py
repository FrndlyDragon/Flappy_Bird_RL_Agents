import torch
from RL.agent import REINFORCE
from train import *
from util import *


if __name__ == "__main__":

    network = 'baseline'
    agent = REINFORCE(network=network, lr=1e-2, epsilon_exploration=False)

    # train
    policy, mean_scores = train(agent, 500)

    # plot
    plot_performance(mean_scores, f'{network}_mean_scores')

    # eval
    eval(agent, policy, n_games=100)

    # save model
    torch.save(policy.state_dict(), f"./models/{network}.pth")

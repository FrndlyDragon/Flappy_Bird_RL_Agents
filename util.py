import os
import matplotlib.pyplot as plt
import numpy as np


def plot_performance(scores, fname, window_size=10):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    steps = np.arange(len(scores))
    window = np.ones(window_size) / window_size  # Normalized averaging kernel
    smoothed_scores = np.convolve(scores, window, mode='valid')

    # Plotting training and validation losses
    plt.plot(steps, scores, label="Original Scores", alpha=0.6)
    plt.plot(steps[window_size-1:], smoothed_scores, label="Smoothed Scores (Moving Average)", linewidth=2)
    plt.title("Game Performance (Moving Average)")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")
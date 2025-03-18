import os
import matplotlib.pyplot as plt


def plot_performance(mean_scores, fname):
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

    # Plotting training and validation losses
    plt.plot(mean_scores, label='Score moving average')
    plt.xlabel('Epoch')
    plt.ylabel('Mean score')
    plt.title('Score moving average vs epoch')

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")
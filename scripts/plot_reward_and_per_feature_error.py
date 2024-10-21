from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp

from pobax.utils.file_system import load_info


import math

def closest_denominators(n):
    # Initialize variables to store the closest pair
    closest_pair = (1, n)
    # Iterate through possible factors from 1 to sqrt(n)
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:  # If i is a factor of n
            # The pair is (i, n // i)
            pair = (i, n // i)
            # Check if this pair is closer than the previous closest pair
            if abs(pair[0] - pair[1]) < abs(closest_pair[0] - closest_pair[1]):
                closest_pair = pair
    return closest_pair


if __name__ == "__main__":
    dataset_path = Path('/Users/ruoyutao/Documents/pobax/results/CartPole-v1_0_4a6d3f8e2b768f4a96c5077e60774023_buffer_2024')
    res_path = dataset_path / 'parsed_errs.npy'
    error_dict = load_info(res_path)
    config = error_dict['config']

    def plot_errs():
        fig, ax = plt.subplots(1, 1)

        diff_per_feat_err = (error_dict['val_err'] - error_dict['R_err'])
        mean_diff_per_feat_err = diff_per_feat_err.mean(axis=-1)
        std_err_diff_per_feat_err = scp.stats.sem(diff_per_feat_err, axis=-1)

        x = np.arange(mean_diff_per_feat_err.shape[0])
        ax.plot(x, mean_diff_per_feat_err, color='blue', label='val_err - R_err')
        ax.fill_between(x, mean_diff_per_feat_err - std_err_diff_per_feat_err, mean_diff_per_feat_err + std_err_diff_per_feat_err,
                          color='blue', alpha=0.35)

        mean_reward_err = error_dict['R_err'].mean(axis=-1)
        std_err_reward_err = scp.stats.sem(error_dict['R_err'], axis=-1)
        ax.plot(x, mean_reward_err, color='orange', label='R_err')
        ax.fill_between(x, mean_reward_err - std_err_reward_err, mean_reward_err + std_err_reward_err,
                        color='orange', alpha=0.35)

        # ax.set_ylim([-0.67, -0.55])
        fig.suptitle(f'{config["ENV_NAME"]}')

        plt.legend(loc='upper right')
        plt.show()

    def plot_feature_mean_and_std():
        mean, std = error_dict['mean_across_timesteps'], error_dict['std_across_timesteps']
        n_features = mean.shape[1]
        n_rows, n_cols = closest_denominators(n_features)
        x = np.arange(mean.shape[0])

        fig, all_axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        cool_cmap = plt.cm.cool
        cool_colors = cool_cmap(np.linspace(0, 1, n_features))

        for i, axes in enumerate(all_axes):
            for j, ax in enumerate(axes):
                feat_num = i * n_rows + j
                feat_mean = mean[:, feat_num]
                feat_std = std[:, feat_num]
                color = cool_colors[feat_num]

                ax.plot(x, feat_mean, color=color, label=f'{feat_num}')
                ax.fill_between(x, feat_mean - feat_std, feat_mean + feat_std,
                                color=color, alpha=0.35)

                ax.set_ylim([-0.1, 1.1])
                if i != (len(all_axes) - 1):
                    ax.set_xticks([])
                if j != 0:
                    ax.set_yticks([])

        fig.suptitle(f'Normalized feature statistics for {config["ENV_NAME"]}')
        fig.tight_layout()

        # plt.legend(loc='upper right')
        plt.show()

    def plot_individual_per_feature_errs():
        pass

    def plot_w_phi():
        weights = error_dict['w_phi']
        n_features = weights.shape[1]
        n_rows, n_cols = closest_denominators(n_features)
        x = np.arange(weights.shape[0])

        fig, all_axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
        cool_cmap = plt.cm.cool
        cool_colors = cool_cmap(np.linspace(0, 1, n_features))

        for i, axes in enumerate(all_axes):
            for j, ax in enumerate(axes):
                feat_num = i * n_rows + j
                color = cool_colors[feat_num]

                ax.plot(x, weights[:, feat_num], color=color, label=f'{feat_num}')

                # ax.set_ylim([-0.1, 1.1])
                # if i != (len(all_axes) - 1):
                #     ax.set_xticks([])
                # if j != 0:
                #     ax.set_yticks([])

        fig.suptitle(f'Feature weights over time for {config["ENV_NAME"]}')
        fig.tight_layout()

        # plt.legend(loc='upper right')
        plt.show()
        # plot_errs()
    plot_w_phi()
    print()

import pickle
from pathlib import Path
from matplotlib import rc
import matplotlib.pyplot as plt

from pobax.utils.plot import mean_confidence_interval

from definitions import ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 24})
rc('axes', unicode_minus=False)


hparam_to_title = {
    'n_action_repeats': 'Num. Action Repeats'
}

env_to_title = {
    'tmaze_10': 'T-Maze 10',
    'CartPole-P-v0': 'Pos. Only CartPole'
}

if __name__ == "__main__":
    hparam = 'n_action_repeats'
    results_dir = Path('/Users/ruoyutao/Documents/pobax/results/tmaze_repeat_sweep/tmaze_10_ppo_repeat_sweep')
    # results_dir = Path('/Users/ruoyutao/Documents/pobax/results/cartpole_repeat_sweep/cartpole_p_ppo_repeat_sweep')

    res = {}
    for res_dir in results_dir.iterdir():
        with open(res_dir / "best_hyperparam_per_env_res_discounted.pkl", "rb") as f:
            best_res = pickle.load(f)
            env = best_res['all_hyperparams']['env']
            scores = best_res['scores'][env].mean(axis=0).mean(axis=0)
            res[best_res['all_hyperparams'][hparam]] = {'scores': scores, 'lambda': best_res['hyperparams'][env]['lambda0']}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for h, res_dict in res.items():
        scores = res_dict['scores']
        lamb = res_dict['lambda']
        m, conf = mean_confidence_interval(scores)
        ax.plot(h, m)
        ax.errorbar(h, m, yerr=conf, fmt='o')
        # ax.annotate(f'lambda = {lamb:.2f}', xy=(h, m), textcoords='data')

    title = f'{hparam_to_title[hparam]} sensitivity in\n{env_to_title[env]}'
    ax.set_title(title)
    ax.set_xlabel(hparam_to_title[hparam])
    ax.set_ylabel(f"Discounted Returns\n({scores.shape[0]} seeds, 95% C.I.)")
    fig.tight_layout()
    plt.show()

    save_plot_to = Path(ROOT_DIR, 'results', f'{title}.pdf')
    fig.savefig(save_plot_to, bbox_inches='tight', dpi=100)
    print(f"Saved figure to {save_plot_to}")


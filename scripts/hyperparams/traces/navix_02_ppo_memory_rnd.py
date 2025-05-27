from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.95]
lambda1s = [0.5]
alphas = [1]
ld_weights = [0]
rnd_reward_coeffs = [0.01, 0.1, 1.0, 10.0, 100.0]
rnd_lrs = [2.5e-4, 2.5e-5, 2.5e-6, 2.5e-7]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo_memory_rnd',
    'args': [
        {
            'env': 'Navix-DMLab-Maze-02-v0',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'rnd_lr': rnd_lrs, 
            'rnd_loss_coeff': 1.0,
            'rnd_gae_coeff': 0.1,
            'rnd_reward_coeff': rnd_reward_coeffs,
            'rnd_hidden_size': 512,
            'normalize_env': True,
            'hidden_size': 512,
            'num_envs': 32,
            'entropy_coeff': 0.01,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(2e7),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
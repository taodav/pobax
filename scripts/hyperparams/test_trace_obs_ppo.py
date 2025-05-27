from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.9]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]
rnd_reward_coeffs = [1.0]
rnd_lrs = [2.5e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': False,
            'action_concat': True,
            'memoryless': True,
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
            'use_trace_features': True,
            'trace_in_obs': True,
            'normalize_env': True,
            'hidden_size': 256,
            'num_envs': 8,
            'entropy_coeff': 0.05,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(5e6),
            'seed': 2024,
            'n_seeds': 3,
            'debug': True,
            'show_discounted': True,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
from pathlib import Path

exp_name = Path(__file__).stem

lrs = [0.00025]
lambda0s = [0.5]
lambda1s = [0.5]
alphas = [1]
ld_weights = [0]
rnd_reward_coeffs = [0.01, 0.1, 1.0, 10.0, 100.0]
rnd_lrs = [2.5e-4, 2.5e-5, 2.5e-6, 2.5e-7]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo_rnd_trace',
    'args': [
        {
            'env': 'craftax',
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
            'normalize_env': True,
            'use_trace_features': True,
            'trace_in_obs': True,
            'normalize_env': True,
            'hidden_size': 512,
            'num_envs': 32,
            'entropy_coeff': 0.01,
            'steps_log_freq': 20,
            'update_log_freq': 16,
            'num_minibatches': 4,
            'num_steps': 64,
            'num_envs': 32,
            'total_steps': int(5e6),
            'seed': 2024,
            'n_seeds': 3,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
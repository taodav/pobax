from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]  # Learning rates
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.95]
lambda1s = [0.95]
cumulant_loss_weights = [0.1, 0.25, 0.5, 0.75]
ld_weights = [0.]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.gd_ppo',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': False,
            'action_concat': True,
            'reward_concat': False,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 256,
            'num_envs': 8,
            'cumulant_type': ['hs', 'obs'],
            'cumulant_transform': ['none', 'random_proj'],
            'cumulant_loss_weight': ' '.join(map(str, cumulant_loss_weights)),
            'add_reward_to_cumulant': True,
            'entropy_coeff': 0.2,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(5e6),
            'sweep_type': 'random',
            'n_random_hparams': 30,
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5]
lambda0s = [0.0]
lambda1s = [0.0]
#alphas = [1]
ld_weights = [0.]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.gd_ppo',
    'args': [
        {
            'env': 'Navix-DMLab-Maze-F-01-v0',
            'double_critic': False,
            'memoryless': False,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 256,
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': lambda1s,
            #'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'entropy_coeff': 0.02,
            'num_steps': 128,
            'num_envs': 32,
            'cumulant_type': 'obs',
            'cumulant_diff': True,
            'cumulant_loss_weight': 0.5,
            'steps_log_freq': 8,
            'update_log_freq': 5,
            'total_steps': int(1e7),
            'seed': 2025,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

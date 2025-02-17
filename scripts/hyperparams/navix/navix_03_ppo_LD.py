from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4, 2.5e-5]  # Learning rates
lambda0s = [0.1, 0.95]
lambda1s = [0.5, 0.7, 0.95]
alphas = [1]
ld_weights = [0.25, 0.5]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'Navix-DMLab-Maze-03-v0',
            'double_critic': True,
            'memoryless': False,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 512,
            'lambda0': lambda0s,
            'lambda1': lambda1s,
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'entropy_coeff': 0.02,
            'num_steps': 128,
            'num_envs': 128,
            'total_steps': int(3e7),
            'seed': 2025,
            'n_seeds': 10,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

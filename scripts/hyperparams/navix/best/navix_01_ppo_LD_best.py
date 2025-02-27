from pathlib import Path

exp_name = Path(__file__).stem

lrs = [0.00025]  # Learning rates
lambda0s = [0.95]
lambda1s = [0.5]
alphas = [1]
ld_weights = [0.25]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'Navix-DMLab-Maze-01-v0',
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
            'entropy_coeff': 0.01,
            'num_steps': 128,
            'num_envs': 256,
            'total_steps': int(1e7),
            'seed': [2126 + i for i in range(10)],
            'n_seeds': 3,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.95]
lambda1s = [0.5]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'Navix-DMLab-Maze-01-v0',
            'double_critic': False,
            'memoryless': False,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 512,
            'lambda0': ' '.join(map(str, lambda1s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'entropy_coeff': 0.1,
            'steps_log_freq': 8,
            'update_log_freq': 16,
            'num_steps': 128,
            'num_envs': 4,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}

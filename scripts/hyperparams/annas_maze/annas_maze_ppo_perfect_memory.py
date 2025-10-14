from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5]
lambda0s = [0.5, 0.7, 0.9, 0.95]
lambda1s = [0.5]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'Navix-Annas-Maze-F-v0',
            'double_critic': False,
            'memoryless': False,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 256,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'entropy_coeff': 0.02,
            'num_steps': 128,
            'num_envs': 64,
            'steps_log_freq': 32,
            'update_log_freq': 20,
            'total_steps': int(5e6),
            'seed': [i + 2025 for i in range(5)],
            'n_seeds': 2,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

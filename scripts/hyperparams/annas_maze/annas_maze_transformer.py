from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.5]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [
        {
            'env': 'Navix-Annas-Maze-v0',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 512,
            'embed_size': 220,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'entropy_coeff': 0.01,
            'num_steps': 128,
            'num_envs': 16,
            'steps_log_freq': 16,
            'update_log_freq': 10,
            'total_steps': int(5e6),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
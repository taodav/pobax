from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3]
lambda0s = [0.7]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'tmaze_10',
            'double_critic': False,
            'action_concat': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 32,
            'num_envs': 4,
            'entropy_coeff': 0.01,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(1e6),
            'seed': 2025,
            'n_seeds': 30,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

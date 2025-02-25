from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3]
lambda0s = [0.3]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [{
        'env': 'rocksample_11_11',
        'double_critic': False,
        'memoryless': True,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': ' '.join(map(str, lambda0s)),
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'num_envs': 8,
        'hidden_size': 256,
        'entropy_coeff': 0.2,
        'steps_log_freq': 4,
        'update_log_freq': 5,
        'total_steps': int(5e6),
        'seed': 2025,
        'n_seeds': 30,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}

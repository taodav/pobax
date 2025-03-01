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
        'env': 'rocksample_15_15',
        'double_critic': False,
        'memoryless': True,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': ' '.join(map(str, lambda0s)),
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'hidden_size': 256,
        'entropy_coeff': 0.2,
        'num_envs': 16,
        'steps_log_freq': 8,
        'update_log_freq': 10,
        'total_steps': int(1e7),
        'seed': 2026,
        'n_seeds': 30,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}

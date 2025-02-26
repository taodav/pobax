from pathlib import Path
import importlib

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.9]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [{
        'env': 'Walker-F-v0',
        'double_critic': False,
        'memoryless': True,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': lambda0s,
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'ld_weight': ' '.join(map(str, ld_weights)),
        'hidden_size': 256,
        'steps_log_freq': 16,
        'update_log_freq': 20,
        'total_steps': int(5e7),
        'seed': 2025,
        'n_seeds': 30,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}

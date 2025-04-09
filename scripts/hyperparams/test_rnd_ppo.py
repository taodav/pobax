from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.5]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]
# rnd_reward_coeffs = [1.0]
rnd_lrs = [2.5e-4]
# rnd_loss_coeffs = [0.01]
# rnd_gae_coeffs = [0.01]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': False,
            'memoryless': False,
            'action_concat': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'rnd_lr': ' '.join(map(str, rnd_lrs)), 
            'hidden_size': 256,
            'num_envs': 8,
            'entropy_coeff': 0.2,
            'rnd_loss_coeff': 0.01,
            'rnd_gae_coeff': 0.01,
            'rnd_reward_coeff': 1.0,
            'rnd_hidden_size': 512,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 1,
            'debug': True,
            'show_discounted': True,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}

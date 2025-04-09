from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4, 2.5e-5]
lambda0s = [0.1, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]
rnd_lrs = [3e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo_memory_rnd',
    'args': [
        {
            'env': 'craftax',
            'double_critic': False,
            'memoryless': True,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 512,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'rnd_lr': ' '.join(map(str, rnd_lrs)), 
            'rnd_loss_coeff': 1.0,
            'rnd_gae_coeff': 0.1,
            'rnd_reward_coeff': 1.0,
            'rnd_hidden_size': 512,
            'entropy_coeff': 0.01,
            'steps_log_freq': 20,
            'update_log_freq': 16,
            'num_minibatches': 4,
            'num_steps': 64,
            'num_envs': 256,
            'total_steps': int(5e8),
            'seed': 2020,
            'n_seeds': 3,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}
"""
To test LD:
check T-Maze optimal policy converges and that values are the same.

To test GVFs:
Have observation signals down a single corridor at different time steps.
The initial GVF should correspond to the discounted time-to-reach each signal

To test hidden-state dependent gammas:
TODO
"""
import jax

from pobax.algos.gd_ppo import GDPPOHyperparams, make_train

def test_ld():
    args = GDPPOHyperparams().from_dict({
        'env': 'fully_observable_simple_chain',
        'gamma': 0.9,
        'hidden_size': 64,
        'cumulant_type': 'rew',
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })
    rng = jax.random.PRNGKey(args.seed)
    make_rng, train_rng, rng = jax.random.split(rng, 3)
    # train_fn = jax.jit(make_train(args, rng))
    train_fn = make_train(args, rng)

    res = train_fn(args.vf_coeff[0], args.ld_weight[0], args.alpha[0], args.lambda1[0], args.lambda0[0], args.lr[0], train_rng)


    pass
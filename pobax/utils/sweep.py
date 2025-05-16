from itertools import product

from tap import Tap
import jax
import jax.numpy as jnp


def get_grid_hparams(args: Tap):
    """
    Will return a set of grid-searchable hyperparams in a v-mappable dictionary,
    where in in_axes is 0.
    :param args: Hyperparams with lists in them
    :return: dictionary of hyperparams to sweep.
    """
    arg_keys, arg_values = [], []
    for arg, val in args.as_dict().items():
        if isinstance(val, list):
            arg_keys.append(arg)
            arg_values.append(val)

    product_arg_values = jnp.array(list(product(*arg_values)))

    hyperparams = {}
    for arg_key, hyperparam_vals in zip(arg_keys, product_arg_values.T):
        hyperparams[arg_key] = hyperparam_vals

    return hyperparams, (arg_keys, product_arg_values)


def get_randomly_sampled_hparams(rng: jax.random.PRNGKey, args: Tap, n_samples: int):
    _, (arg_keys, all_product_arg_vals) = get_grid_hparams(args)
    chosen_idxes = jax.random.choice(rng, all_product_arg_vals.shape[0], shape=(n_samples,), replace=False)

    chosen_product_arg_vals = all_product_arg_vals[chosen_idxes]

    hyperparams = {}
    for arg_key, hyperparam_vals in zip(arg_keys, chosen_product_arg_vals.T):
        hyperparams[arg_key] = hyperparam_vals

    return hyperparams


if __name__ == "__main__":
    from pobax.config import PPOHyperparams
    args = PPOHyperparams().parse_args()
    args_dict, _ = get_grid_hparams(args)

    rng = jax.random.PRNGKey(2025)
    chosen_args_dict = get_randomly_sampled_hparams(rng, args, n_samples=5)

    print()

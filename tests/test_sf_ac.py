from gymnax.environments import spaces
import jax.numpy as jnp
import jax

from pobax.models.network import ScannedRNN
from pobax.models.actor_critic import SFActorCritic


def test_grads():
    """
    Test that we have zero gradients for reward params in sf_loss,
    as well as zero gradients for all non-reward parameters for reward_loss.
    """
    rng = jax.random.PRNGKey(2025)

    network = SFActorCritic(spaces.Discrete(3),
                            hidden_size=8,
                            memoryless=True)

    init_x = (
        jnp.ones(
            (1, 4, 12)
        ),
        jnp.zeros((1, 4)),
    )
    init_hstate = ScannedRNN.initialize_carry(4, 8)

    network_params = network.init(rng, init_hstate, init_x)


    def sf_loss(params, hstate, x):
        rew_params = network_params['params']['r']
        rew_params = (rew_params['kernel'], rew_params['bias'])
        _, _, v, encoding = network.apply(params, hstate, x, rew_params, method=SFActorCritic.get_sf)
        return (v ** 2).sum(), encoding


    sf_grad_fn = jax.value_and_grad(sf_loss, has_aux=True)
    vals, sf_grads = sf_grad_fn(network_params, init_hstate, init_x)

    # Check that all reward grads are 0
    reward_grads = jax.tree.flatten(sf_grads['params']['r'])[0]
    for g in reward_grads:
        assert jnp.allclose(g, 0)

    reward_grads = sf_grads['params']['r']

    _, encoding = vals


    def reward_loss(params, encoding):
        r = network.apply(params, encoding, method=SFActorCritic.get_reward)
        return (r ** 2).sum()


    # encoding = network.apply(network_params, init_x[0], method=SFActorCritic.get_encoding)
    reward_grad_fn = jax.value_and_grad(reward_loss)
    vals, rew_grads = reward_grad_fn(network_params, encoding)

    for k, v in rew_grads['params'].items():
        if k != 'r':
            for g in jax.tree.flatten(v)[0]:
                assert jnp.allclose(g, 0)


if __name__ == "__main__":
    test_grads()

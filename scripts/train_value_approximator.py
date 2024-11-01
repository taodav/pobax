from collections import namedtuple
from pathlib import Path
from typing import Union
from typing import Literal
import flax.linen as nn
from jax._src.nn.initializers import orthogonal, constant

from flax.training import orbax_utils
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
import optax
import orbax.checkpoint

from pobax.models.network import SimpleNN, ScannedRNN, RNNApproximator
from pobax.models.value import Critic
from pobax.replay import make_flat_buffer
from pobax.replay.trajectory import TrajectoryBufferState
from pobax.utils.file_system import make_hash_md5, load_info, numpyify_and_save, load_train_state
import matplotlib.pyplot as plt

from definitions import ROOT_DIR

class ValueHyperparams(Tap):
    dataset_path: Union[str, Path]
    approximator: Literal['mlp', 'rnn_skip'] = 'mlp'
    target: Literal['td', 'mc'] = 'td'

    hidden_size: int = 128
    n_hidden_layers: int = 3
    lr: float = 1e-4

    epochs: int = 100
    train_steps: int = int(1e6)
    batch_size: int = 32

    study_name: str = 'test'
    debug: bool = True
    seed: int = 2024
    platform: str = 'gpu'

    def configure(self) -> None:
        self.add_argument('--dataset_path', type=Path)

class ValueNetwork(nn.Module):
    approximator: str
    hidden_size: int = 128
    n_hidden_layers: int = 3

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if self.approximator == 'rnn_skip':
            embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)
            rnn_in = (embedding, dones)
            _, embedding = RNNApproximator(hidden_size=self.hidden_size, horizon=self.n_hidden_layers)(hidden, rnn_in)
        else:
            embedding = SimpleNN(hidden_size=self.hidden_size, depth=self.n_hidden_layers)(obs)

        critic = Critic(hidden_size=self.hidden_size)

        v = critic(embedding)

        return jnp.squeeze(v, axis=-1)

def make_train(args: ValueHyperparams):
    args.steps_per_epoch = args.train_steps // args.epochs
    args.dataset_path = Path(args.dataset_path).resolve()

    # load dataset
    if args.dataset_path.suffix == '.npy':
        restored = load_info(args.dataset_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(args.dataset_path)

    # dataset should be a dict of ['observation', 'action', 'reward', 'done', 'next_observation', 'V', 'G']
    dataset = restored['dataset']
    experience = jax.tree.map(lambda x: jnp.array(x), dataset)

    Experience = namedtuple('Experience', list(experience.keys()))
    experience = Experience(**experience)

    network = ValueNetwork(approximator=args.approximator, hidden_size=args.hidden_size, n_hidden_layers=args.n_hidden_layers)

    experience_size = experience.observation.shape[0]
    buffer = make_flat_buffer(
        max_length=experience_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    buffer_state = TrajectoryBufferState(
        current_index=jnp.array(0, dtype=int),
        is_full=jnp.array(True),
        experience=jax.tree_util.tree_map(lambda x: x[None, ...], experience)
    )

    def train(rng):
        params_rng, rng = jax.random.split(rng)
        ac_in = (jnp.expand_dims(experience.observation[:1], axis=0), jnp.expand_dims(experience.done[:1], axis=0))
        init_hidden = ScannedRNN.initialize_carry(1, args.hidden_size)
        params = network.init(params_rng, init_hidden, ac_in)
        tx = optax.adam(args.lr, eps=1e-5)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        def _epoch_step(runner_state, i):
        # @scan_tqdm(args.train_steps, print_rate=100)
        #     @jax.jit
            def _update_step(runner_state, i):
                ts, rng = runner_state

                sample_key, rng = jax.random.split(rng)
                batch = buffer.sample(buffer_state, sample_key)

                def _compute_target(params, batch, target_key):
                    if target_key == 'mc':
                        target = getattr(batch.experience.first, 'G').astype(float)
                    else:
                        reward = getattr(batch.experience.first, 'reward').astype(float)
                        done = getattr(batch.experience.first, 'done').astype(float)
                        ac_in = (jnp.expand_dims(batch.experience.first.next_observation, axis=1), jnp.expand_dims(batch.experience.first.done, axis=1))
                        next_value = network.apply(params, init_hidden, ac_in)
                        next_value = next_value.squeeze(axis=-1)
                        target = reward + (1 - done) * next_value
                    return target
                
                target = _compute_target(ts.params, batch, args.target)

                def _loss_fn(params: dict):
                    ac_in = (jnp.expand_dims(batch.experience.first.observation, axis=1), jnp.expand_dims(batch.experience.first.done, axis=1))
                    value = network.apply(params, init_hidden, ac_in)
                    value = value.squeeze(axis=-1)
                    loss = optax.l2_loss(value, target).sum(axis=-1)
                    return loss.mean()

                grad_fn = jax.jit(jax.value_and_grad(_loss_fn))
                loss, grads = grad_fn(ts.params)
                new_ts = ts.apply_gradients(grads=grads)
                return (new_ts, rng), loss

            runner_state, losses = jax.lax.scan(
                _update_step, runner_state, jnp.arange(args.steps_per_epoch), args.steps_per_epoch
            )
            if args.debug:
                jax.debug.print("Step {step} average loss: {loss}", step=(i * args.steps_per_epoch), loss=losses.mean())

            return runner_state, losses.mean()

        runner_state = (train_state, rng)
        runner_state, epoch_losses = jax.lax.scan(
            _epoch_step, runner_state, jnp.arange(args.epochs), args.epochs
        )
        return {
            'final_train_state': runner_state[0], 
            'epoch_losses': epoch_losses, 
            'args': args.as_dict(),
        }

    return train


if __name__ == '__main__':
    # jax.disable_jit(True)
    args = ValueHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    train_key, key = jax.random.split(key)

    train_fn = make_train(args)
    # train_fn = jax.jit(train_fn)

    out = train_fn(train_key)

    out['args'] = args.as_dict()

    def path_to_str(d: dict):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                path_to_str(v)
    path_to_str(out)

    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"value_distance_{args.approximator}_{args.target}"

    print(f"Saving results to {results_path}")
    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(out)
    orbax_checkpointer.save(results_path, out, save_args=save_args)
    # numpyify_and_save(results_path, out)

    print("Done.")
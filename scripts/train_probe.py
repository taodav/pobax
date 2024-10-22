from collections import namedtuple
from pathlib import Path
from typing import Union

from flax.training import orbax_utils
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
import optax
import orbax.checkpoint

from pobax.models.network import ProbePredictorNN
from pobax.replay import make_flat_buffer
from pobax.replay.trajectory import TrajectoryBufferState
from pobax.utils.file_system import make_hash_md5, load_info, numpyify_and_save
import matplotlib.pyplot as plt

from definitions import ROOT_DIR


class ProbeHyperparams(Tap):
    dataset_path: Union[str, Path]
    input_key: str = 'memoryless_embedding'  # Key in dataset that we set as input
    target_key: str = 'observation'  # Key in dataset that we set as target

    hidden_size: int = 512
    n_hidden_layers: int = 2
    lr: float = 1e-4

    epochs: int = 100
    train_steps: int = int(4e6)
    batch_size: int = 32

    study_name: str = 'test'
    debug: bool = True
    seed: int = 2024
    platform: str = 'gpu'

    def configure(self) -> None:
        self.add_argument('--dataset_path', type=Path)

def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]

def make_train(args: ProbeHyperparams):
    args.steps_per_epoch = args.train_steps // args.epochs
    args.dataset_path = Path(args.dataset_path).resolve()

    if args.dataset_path.suffix == '.npy':
        restored = load_info(args.dataset_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(args.dataset_path)

    dataset_args, dataset = restored['args'], restored['dataset']
    experience = jax.tree.map(lambda x: jnp.array(x), dataset)
    if 'x' not in experience:
        experience['x'] = experience[f'x_{args.input_key}']
    for key in list(experience.keys()):
        if key.startswith('x_'):
            del experience[key]

    # normalize the target
    all_targets = experience[args.target_key]
    target_mean = all_targets.mean(axis=0)
    target_std = all_targets.std(axis=0) + 1e-8
    experience[args.target_key] = (experience[args.target_key] - target_mean) / target_std
    print(f'experience[args.target_key].shape: {experience[args.target_key].shape}')

    Experience = namedtuple('Experience', list(experience.keys()))
    experience = Experience(**experience)

    n_predictions = getattr(experience, args.target_key).shape[-1]
    args.n_predictions = n_predictions

    network = ProbePredictorNN(hidden_size=args.hidden_size,
                                n_outs=n_predictions,
                                n_hidden_layers=args.n_hidden_layers)

    experience_size = experience.x.shape[0]
    buffer = make_flat_buffer(
        max_length=experience_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        # add_batch_size=experience_size
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
        params = network.init(params_rng, experience.x[:1])
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

                target = getattr(batch.experience.first, args.target_key).astype(float)

                def _loss_fn(params: dict):
                    _, logits = network.apply(params, batch.experience.first.x)
                    mask = jnp.ones_like(logits)
                    mask = mask.at[:, 0:2].set(0)

                    # Apply the mask by setting indices 4 and 5 to zero in logits and target
                    masked_logits = logits * mask
                    masked_target = target * mask
                    loss = optax.l2_loss(masked_logits, masked_target).sum(axis=-1)
                    # loss = optax.l2_loss(logits, target).sum(axis=-1)
                    return loss.mean()

                grad_fn = jax.jit(jax.value_and_grad(_loss_fn))
                loss, grads = grad_fn(ts.params)
                new_ts = ts.apply_gradients(grads=grads)
                return (new_ts, rng), loss

        # runner_state = (train_state, rng)
        # runner_state, losses = jax.lax.scan(
        #     _update_step, runner_state, jnp.arange(args.train_steps), args.train_steps
        # )
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
            # 'final_train_state': runner_state[0], 
            'epoch_losses': epoch_losses, 
            'args': args.as_dict(),
            # 'ckpt': restored['ckpt']
        }

        # return {
        #     'final_train_state': runner_state[0], 'losses': losses
        # }
    return train


if __name__ == '__main__':
    # jax.disable_jit(True)
    args = ProbeHyperparams().parse_args()
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

    results_dir = Path(ROOT_DIR, 'results', f'{args.input_key}')
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"probe"

    print(f"Saving results to {results_path}")
    # Save all results with Orbax
    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # save_args = orbax_utils.save_args_from_target(out)
    # orbax_checkpointer.save(results_path, out, save_args=save_args)
    numpyify_and_save(results_path, out)

    print("Done.")

    # epoch_losses = out['epoch_losses']

    # # Plot the epoch losses
    # plt.figure(figsize=(10, 6))
    # plt.plot(epoch_losses, label='Epoch Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Epochs')
    # plt.legend()
    # plt.grid(True)

    # # Save the plot
    # plot_path = results_dir / f"{args.target_key}_seed_{args.seed}_features_idx_{args.features_idx}_epoch_loss.png"
    # plt.savefig(plot_path)
    # print(f"Epoch loss plot saved to {plot_path}")

    # # Display the plot
    # plt.show()

    # print("Done.")
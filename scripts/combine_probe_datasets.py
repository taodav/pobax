from pathlib import Path

from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from typing import Union
from tap import Tap
from pobax.utils.file_system import numpyify_and_save, load_info
from definitions import ROOT_DIR

class CombineHyperparams(Tap):
    dataset_0_path: Union[str, Path]
    dataset_1_path: Union[str, Path]

    update_idx_to_take: int = None

    num_envs: int = 4
    n_samples: int = int(1e6)

    seed: int = 2024
    platform: str = 'gpu'
    study_name: str = 'test'

    def configure(self) -> None:
        self.add_argument('--dataset_0_path', type=Path)
        self.add_argument('--dataset_1_path', type=Path)

if __name__ == "__main__":
    args = CombineHyperparams().parse_args()
    jax.config.update('jax_platform_name', 'gpu')
    
    d0_path = Path(args.dataset_0_path).resolve()
    d1_path = Path(args.dataset_1_path).resolve()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    if args.dataset_0_path.suffix == '.npy':
        restored = load_info(d0_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(d0_path)

    args_0, ckpt_0, dataset_0 = restored['args'], restored['ckpt'], restored['dataset']

    if args.dataset_1_path.suffix == '.npy':
        restored = load_info(d1_path)
    else:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(d1_path)

    args_1, ckpt_1, dataset_1 = restored['args'], restored['ckpt'], restored['dataset']

    print(f"Combining datasets {d0_path} and {d1_path}")

    combined_dataset = jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), dataset_0, dataset_1)

    save_dir = Path(ROOT_DIR, 'results', 'combined_probe_datasets')
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'combined_probe_datasets.npy'
    to_save = {
        'args': [args_0, args_1],
        'ckpt': [ckpt_0, ckpt_1],
        'dataset': combined_dataset
    }

    print(f"Saving results to {save_path}")
    numpyify_and_save(save_path, to_save)

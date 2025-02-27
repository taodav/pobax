from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
import os
# os.environ["CRAFTAX_RELOAD_TEXTURES"] = "True"

# Set the MUJOCO_GL environment variable to use EGL
# os.environ['MUJOCO_GL'] = 'egl'

from pobax.envs import get_env
class CollectHyperparams(Tap):
    env: str = 'craftax_pixels'

    update_idx_to_take: int = None

    num_envs: int = 4
    n_samples: int = int(1e3)
    gamma: float = 0.99
    image_size: int = 64

    seed: int = 2024
    platform: str = 'gpu'
    study_name: str = 'test'

def ppo_step(runner_state, unused, *, env, num_envs):
    """
    One environment step under a random policy.

    runner_state: (obs, done, env_state, rng)
    """
    # Unpack current runner state
    obs, done, env_state, rng, env_params = runner_state

    @jax.jit
    def sample_action(rng):
        # Split RNG for each environment to sample random actions
        rng, rng_action = jax.random.split(rng)
        rng_action = jax.random.split(rng_action, num_envs)

        # Vectorized sampling
        sample_fn = jax.vmap(env.action_space(env_params).sample)
        actions = sample_fn(rng_action)
        return actions, rng

    # Sample random actions from the environment's action space
    actions, rng = sample_action(rng)
    
    # Step the environment
    rng, rng_step = jax.random.split(rng)
    next_obs, next_env_state, reward, next_done, info = env.step(
        rng_step, env_state, actions, env_params
    )
    # Collect data from this step (e.g., for replay buffers, logging, etc.)
    datum = {
        'frame': next_obs,  # For rendering later
        # you can store reward, done, etc. if desired
    }

    # Update runner state
    runner_state = (next_obs, next_done, next_env_state, rng, env_params)
    return runner_state, datum

def make_collect(args: CollectHyperparams, key: chex.PRNGKey):
    steps_to_collect = args.n_samples // args.num_envs

    rng = jax.random.PRNGKey(args.seed)
    env, env_params = get_env(args.env, rng, normalize_image=False)

    _env_step = partial(ppo_step, num_envs=args.num_envs,
                        env=env)
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    def collect(rng):
        # INIT ENV
        rng, reset_keys = jax.random.split(rng)
        obsv, env_states = env.reset(reset_keys, env_params)

        # init hidden state
        runner_state = (
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            env_states,
            rng,
            env_params
        )

        runner_state, dataset = jax.lax.scan(
            _env_step, runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        return dataset

    return collect

def make_video(args: CollectHyperparams, frame_data: dict, max_frames: int = 1000):
    """
    Create a video from collected frames of multiple environments arranged in a grid.

    Args:
        args (CollectHyperparams): Configuration parameters.
        frame_data (dict): Collected frame data with key 'frame'.
        max_frames (int): Maximum number of frames to include in the video.
    """
    import cv2
    from pathlib import Path

    # Extract frames from the dataset
    frames = frame_data['frame']  # Shape: (2500, 4, 64, 64, 4)
    frames = np.array(frames)

    # Limit the number of frames to avoid excessively large videos
    frames = frames[:max_frames]  # Shape: (max_frames, 4, 64, 64, 4)

    # Create a directory to save the video
    save_path = Path('./videos')
    save_path.mkdir(parents=True, exist_ok=True)
    video_filename = save_path / f'{args.env}_random_policy.mp4'

    # Determine individual frame size and grid size
    individual_height, individual_width, individual_channels = frames[0, 0].shape  # (64, 64, 4)
    grid_rows, grid_cols = 1, args.num_envs  # 2x2 grid for 4 environments

    # Calculate grid frame size
    grid_height = individual_height * grid_rows
    grid_width = individual_width * grid_cols

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(str(video_filename), fourcc, 30.0, (grid_width, grid_height))

    for i in range(frames.shape[0]):
        # Initialize empty grid frame
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for env_idx in range(args.num_envs):
            # Calculate position in grid
            row = env_idx // grid_cols
            col = env_idx % grid_cols

            # Extract individual frame
            frame = frames[i, env_idx, :, :, :]  # Shape: (64, 64, 4)
            frame = frame
            # Handle alpha channel: Convert RGBA to BGR by discarding alpha
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Discard alpha channel

            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Place the frame in the grid
            start_y = row * individual_height
            start_x = col * individual_width
            grid_frame[start_y:start_y + individual_height, start_x:start_x + individual_width, :] = frame_bgr

        # Write the grid frame to the video
        out.write(grid_frame)

        # Optional: Print progress
        if (i + 1) % 1000 == 0:
            print(f'Processed {i + 1}/{frames.shape[0]} frames')

    # Release the video writer
    out.release()
    print(f'Video saved to {video_filename}')


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = CollectHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    make_key, collect_key, key = jax.random.split(key, 3)

    collect_fn = make_collect(args, make_key)
    collect_fn = jax.jit(collect_fn)

    frame = collect_fn(collect_key)
    print(frame['frame'].shape)
    video = make_video(args, frame)

    print("Done.")

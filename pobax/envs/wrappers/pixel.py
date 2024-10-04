from functools import partial

from brax import base
import chex
from flax import struct
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp

from pobax.envs.jax.render import (
    get_camera, get_target, render_instances,
    _build_objects, _with_state,
    Camera, Obj)
from .gymnax import GymnaxWrapper


@struct.dataclass
class RenderState:
    camera: Camera
    objects: list[Obj]
    targets: jnp.ndarray


@struct.dataclass
class PixelBraxEnvState:
    render: RenderState
    env_state: environment.EnvState


class PixelBraxEnv(GymnaxWrapper):
    """
    Visual Vector Environment for the Brax API.
    Isn't fully JIT-able, since rendering requires non-JITable functions.
    """
    def __init__(self, env, n_frame_stack: int = 1,
                 size: int = 64):
        super().__init__(env)
        self.n_frame_stack = n_frame_stack
        self.size = size
        self.render_instances_of_size = partial(render_instances, width=size, height=size, enable_shadow=True)

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.size, self.size, 3),
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)

        brax_state, sys = state.pipeline_state, self._unwrapped._env.sys
        camera = get_camera(sys, brax_state)
        target = get_target(brax_state)
        objs = _build_objects(sys)
        render_state = RenderState(camera, objs, target)

        state = PixelBraxEnvState(env_state=state, render=render_state)

        instance = _with_state(brax_state, brax_state.x.concatenate(base.Transform.zero((1,))))
        img = self.render_instances_of_size(instance)
        return img, state


    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params=params)


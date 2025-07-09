from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Tuple
import json



@chex.dataclass
class LightBulbsState:
    bulbs: chex.Array # current state of lightbulbs array.
    goal: chex.Array # The goal array. 
    t: chex.Array # step counter. 


class LightBulbs(Environment):


    def __init__(self,
                 size: int,
                 config_path: str, 
                 ): 
        """
        size: How many light bulbs in the array.
        config_path: path to json with keys "goals" and "distribution" containing array of possible goals and likelihood of choosing each goal. 
        """

        self.size = size
        self.goals, self.goal_distribution = self._load_goals_from_json(config_path, size)

    def _load_goals_from_json(self, path: str, size: int):
        with open(path, 'r') as f:
            data = json.load(f)

        goals = jnp.array(data["goals"], dtype=jnp.int32)
        distribution = jnp.array(data["distribution"], dtype=jnp.float32)

        if goals.shape[1] != size:
            raise ValueError(f"Expected goal arrays of length {size}, but got {goals.shape[1]}.")

        if distribution.shape[0] != goals.shape[0]:
            raise ValueError(f"Distribution length {distribution.shape[0]} does not match number of goals {goals.shape[0]}.")

        distribution = distribution / jnp.sum(distribution)  # ensure normalized

        return goals, distribution



    def observation_space(self, env_params: EnvParams):
        """
        An observation is a binary array of size `self.size`.
        So we will use Box with low = 0 and high = 1 and dimensions (size,).
        """
        return gymnax.environments.spaces.Box(0, 1, (self.size,))
        

    def action_space(self, env_params: EnvParams): #TODO Why do the action and observations spaces take EnvParams as inputs?
        """
        An action is a choice of index to toggle.
        So an action is one of `self.size` many discrete options. 
        """
        return gymnax.environments.spaces.Discrete(self.size)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: LightBulbsState) -> jnp.ndarray:
        '''
        The robot only oberves the state of the bulbs array.

        Returns:
        Current state of bulbs array, i.e. `LightBulbState.bulbs`
        '''
        # TODO it might learn better if it sees the human action. 
        return state.bulbs
    
    @staticmethod
    def human_policy(key: jax.random.key, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        """
        Flip exactly one randomly-chosen bit where `state` and `goal` differ.
        If they already match, return `state` unchanged.
        #TODO This is mostly chatGPT, understand each line later. 
        """
        mask = state != goal                 # boolean mask: True where mismatch
        has_mismatch = jnp.any(mask)         # scalar bool

        def _flip(_):
            # Build a probability vector over *all* indices: 1 for mismatches, 0 otherwise.
            probs = mask.astype(jnp.float32)
            probs /= jnp.sum(probs)          # normalise → probs sums to 1
            idx = random.choice(key, state.shape[0], p=probs)
            return state.at[idx].set(goal[idx])

        # If no mismatches, just return `state`
        new_state = jax.lax.cond(has_mismatch,
                                _flip,                 # true branch
                                lambda _: state,       # false branch
                                operand=None)

        return new_state


    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, LightBulbsState]:
        """
        Choose a random starting array and call it `bulbs`.
        Use self.goal_set and self.goal_distribution to choose a goal array and call it `goal`.
        Returns:
        Tuple of
        1. first observation
        2. LightBulbsState with the 3 attributes above and timestep counter set at 0.
        """

        def _random_binary_array(key, size):
            return jax.random.bernoulli(key, p=0.5, shape=(size,)).astype(jnp.int32)
        
        def _sample_theta(key, distribution):
            return jax.random.choice(key, distribution.shape[0], p=distribution)

        key, subkey = jax.random.split(key)
        bulbs = _random_binary_array(subkey, self.size)

        key, subkey = jax.random.split(key)
        theta = _sample_theta(subkey, self.goal_distribution)

        goal = self.goals[theta]

        state = LightBulbsState(bulbs = bulbs,
                                goal = goal,
                                t = 0)
        
        return self.get_obs(state), state
    
    def transition(self, key: chex.PRNGKey, state: LightBulbsState, action: int):
        """
        Steps:
        1. Toggle the bulb at the given action index.
        2. Have the human policy toggle one bit toward the goal.
        3. Check if the resulting bulb state matches the goal.
        4. Compute reward.
        """

        # 1. Toggle the robot's chosen switch
        bulbs_after_robot = state.bulbs.at[action].set(1 - state.bulbs[action])

        # 2. Human responds: flip one mismatching bit
        key, subkey = jax.random.split(key)
        bulbs_after_human = self.human_policy(subkey, bulbs_after_robot, state.goal)

        # 3. Check for goal match
        done = jnp.all(bulbs_after_human == state.goal)

        # 4. Reward
        reward = jnp.where(done, 0, -1)

        # 5. Increment timestep
        next_t = state.t + 1

        # 6. Create new state
        next_state = LightBulbsState(
            bulbs=bulbs_after_human,
            goal=state.goal,
            t=next_t
        )

        return next_state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey, # TODO don't really understand if this key is necessary. 
                 state: LightBulbsState,
                 action: int,
                 params: EnvParams):
    
        key, subkey = jax.random.split(key) #TODO I'm worried that I might be using the same key time and time again.
        next_state, reward, done = self.transition(subkey, state, action) 
        next_observation = next_state.bulbs
        return next_observation, next_state, reward, done, {}


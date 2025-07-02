from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple


@chex.dataclass
class LightBulbsState:
    bulbs: chex.Array # current state of lightbulbs array.
    goal: chex.Array # The goal array. 
    human_policy: chex.Array # The human policy for the episode. 
    # TODO: Is it reasonable to have this be part of the state?
    # I think the human policy should be an [action_space x observation_space] matrix that specifies human 
    # action probabilities for each possible bulbs array. 
    t: chex.Array # step counter. 


class LightBulbs(Environment):


    def __init__(self,
                 size: int,
                 goal_set: None, #TODO What should be the type?,
                 goal_distribution: jnp.ndarray, 
                 function_build_human_policy : None #TODO What should be the type?
                 ): 
        # TODO not exactly sure about this because I'm not sure if init runs at the start of each episode or at the start of the experiment. 
        """
        size: How many light bulbs in the array.
        goal_set: Set of binary arrays that might be chosen as a goal. 
        goal_distribution: Array of size len(goal_set) with the likelihood of choosing each goal.
        function_build_human_policy: a function that would take a specific goal and output a human model that is attempting to reach that goal.
        """


    def observation_space(self, env_params: EnvParams):
        """
        An observation is a binary array of size `self.size`.
        So we will use Box with low = 0 and high = 1 and dimensions (size,).
        """
        # TODO BTW I'm not including the latest human action as an explicit part of the observation
        # because the last human action is already implicit in the transition and this way it seems like it'd be easier to code. 
        # Is that reasonable? 
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
        # TODO We've mentioned that in the theoretical framework the robot has access to the set of goals and a prior over it. 
        # In this implementation sketch the robot sort of has access to the set of goals in that eventually it gets reward when
        # it reaches a specific bulbs arrangement. 
        # But should it have like literal access to the set of goals? 
        # Would that be part of this get_obs function? 

        return LightBulbsState.bulbs
    
    def build_human_policy(goal: jnp.ndarray) -> jnp.ndarray:
        """
        For a given goal (binary array) build a human policy that attempts that goal.  
        For now the human policy I want is a policy that chooses a random lightbulb that needs fixing and fixes it.
    
        Returns:
        An action_space x observation_space that specifies how likely an action is to be chosen given a specific observation. 
        """
        return 

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, LightBulbsState]:
        """
        Choose a random starting array and call it `bulbs`.
        Use self.goal_set and self.goal_distribution to choose a goal array and call it `goal`.
        Use self.function_build_human_policy to build a human policy for that goal array.

        Returns:
        Tuple of
        1. first observation
        2. LightBulbsState with the 3 attributes above and timestep counter set at 0.
        """

        state = LightBulbsState(bulbs = bulbs,
                                human_policy=human_policy,
                                goal = goal,
                                t = 0)
        
        return self.get_obs(state), state
    
    def transition(self, key: chex.PRNGKey, state: LightBulbsState, action : int): #TODO Does it need actually need this key as input? 
        """
        Steps:
        1. toggle the switch at index given by `action`
        2. then toggle the switch that `LightBulbState.human_policy` in that new state. 
        3. Make boolean value `done` True if the new LightBulbState.bulbs coincides with LightBulbState.goal. 

        Returns:
        1. The new LightBulbState after the two toggles described above.
        2. reward (let's do -0.1 for every step that doesn't reach the goal and then 1 if it does reach the goal). 
        3. the boolean `done` describe above. 
        """
        next_state, reward, done = None, None, None # TODO 
        return next_state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey, # TODO don't really understand if this key is necessary. 
                 state: LightBulbsState,
                 action: int,
                 params: EnvParams):
        next_state, reward, done = self.transition(key, state, action) # TODO should it return a new random key?
        next_observation = next_state.bulbs
        return next_observation, next_state, reward, done, {}


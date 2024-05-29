# Taken from https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py
import jax
import chex

import pobax.mcts.tree as tree_lib
from pobax.mcts import base


def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> chex.Array:
    return jax.lax.cond(
        depth == 0,
        lambda x: root_action_selection_fn(*x[:3]),
        lambda x: interior_action_selection_fn(*x),
        (rng_key, tree, node_index, depth))

  return switching_action_selection_fn

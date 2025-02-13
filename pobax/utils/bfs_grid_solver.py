import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def jax_shortest_path_policy(grid, start, start_orientation, goal):
    """
    Compute the shortest path policy using a BFS over an augmented state space,
    in a JAX–jit-compilable way.

    Parameters:
      grid: jnp.array of shape (H, W) with 0 for free cells and 1 for walls.
      start: tuple (row, col) indicating start position.
      start_orientation: int in {0,1,2,3} (0: up, 1: right, 2: down, 3: left).
      goal: tuple (row, col) for the goal position.

    Returns:
      A tuple (policy, count) where policy is a jnp.array of actions (of fixed size)
      and count is the number of valid actions. Outside of JIT you can then do:

          policy_valid = policy[:count]

      If no path is found, count will be 0.
    """
    H, W = grid.shape
    max_states = H * W * 4  # maximum number of states

    # Convert inputs to jnp arrays / int32
    start = jnp.array(start, dtype=jnp.int32)
    goal = jnp.array(goal, dtype=jnp.int32)
    start_orientation = jnp.int32(start_orientation)

    # Fixed-size queue: each entry is a state [r, c, o]
    queue = -jnp.ones((max_states, 3), dtype=jnp.int32)
    queue = queue.at[0].set(jnp.array([start[0], start[1], start_orientation], dtype=jnp.int32))
    q_start = jnp.array(0, dtype=jnp.int32)
    q_end = jnp.array(1, dtype=jnp.int32)

    # visited[r,c,o]: whether state (r,c,o) has been reached.
    visited = jnp.zeros((H, W, 4), dtype=bool)
    visited = visited.at[start[0], start[1], start_orientation].set(True)

    # For each state we record:
    #  - parent[r,c,o]: flattened index of parent state (or -1 if none)
    #  - action_taken[r,c,o]: action taken to get to (r,c,o)
    parent = -jnp.ones((H, W, 4), dtype=jnp.int32)
    action_taken = -jnp.ones((H, W, 4), dtype=jnp.int32)

    found = False
    found_state = jnp.array([-1, -1, -1], dtype=jnp.int32)

    iter_init = jnp.array(0, dtype=jnp.int32)

    # BFS state is a tuple:
    # (q_start, q_end, queue, visited, parent, action_taken, found, found_state, iter)
    def cond_fun(state):
        q_start, q_end, queue, visited, parent, action_taken, found, found_state, iter_val = state
        return jnp.logical_and(jnp.logical_and(~found, q_start < q_end),
                               iter_val < max_states)

    def body_fun(state):
        q_start, q_end, queue, visited, parent, action_taken, found, found_state, iter_val = state
        current = queue[q_start]  # current state: [r, c, o]
        r, c, o = current[0], current[1], current[2]
        q_start_new = q_start + 1

        # Process all three actions.
        def process_action(i, inner_state):
            q_end_inner, visited_inner, parent_inner, action_taken_inner, queue_inner, found_inner, found_state_inner = inner_state

            # Compute new state based on action i.
            def forward_action():
                dr = jnp.where(o == 0, -1,
                               jnp.where(o == 1, 0,
                                         jnp.where(o == 2, 1, 0)))
                dc = jnp.where(o == 0, 0,
                               jnp.where(o == 1, 1,
                                         jnp.where(o == 2, 0, -1)))
                return r + dr, c + dc, o

            def left_action():
                new_o = (o - 1) % 4
                return r, c, new_o

            def right_action():
                new_o = (o + 1) % 4
                return r, c, new_o

            new_r, new_c, new_o = lax.switch(i, [forward_action, left_action, right_action])
            valid = lax.cond(i == 0,
                             lambda _: ((new_r >= 0) & (new_r < H) &
                                        (new_c >= 0) & (new_c < W) &
                                        (grid[new_r, new_c] == 0)),
                             lambda _: True,
                             operand=None)
            not_visited = ~visited_inner[new_r, new_c, new_o]
            new_valid = valid & not_visited

            def update_fn(_):
                visited_updated = visited_inner.at[new_r, new_c, new_o].set(True)
                parent_idx = ((r * W + c) * 4 + o)
                parent_updated = parent_inner.at[new_r, new_c, new_o].set(parent_idx)
                action_updated = action_taken_inner.at[new_r, new_c, new_o].set(i)
                new_state = jnp.array([new_r, new_c, new_o], dtype=jnp.int32)
                queue_updated = queue_inner.at[q_end_inner].set(new_state)
                q_end_updated = q_end_inner + 1
                reached = (new_r == goal[0]) & (new_c == goal[1])
                found_updated = found_inner | reached
                found_state_updated = lax.select(reached, new_state, found_state_inner)
                return (q_end_updated, visited_updated, parent_updated,
                        action_updated, queue_updated, found_updated, found_state_updated)

            result = lax.cond(new_valid,
                              update_fn,
                              lambda _: (q_end_inner, visited_inner, parent_inner,
                                         action_taken_inner, queue_inner, found_inner, found_state_inner),
                              operand=None)
            return result

        init_inner = (q_end, visited, parent, action_taken, queue, found, found_state)
        new_inner = lax.fori_loop(0, 3, process_action, init_inner)
        q_end_new, visited_new, parent_new, action_taken_new, queue_new, found_new, found_state_new = new_inner
        return (q_start_new, q_end_new, queue_new, visited_new, parent_new,
                action_taken_new, found_new, found_state_new, iter_val + 1)

    init_state = (q_start, q_end, queue, visited, parent, action_taken,
                  found, found_state, iter_init)
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, _, queue, visited, parent, action_taken, found, found_state, _ = final_state

    # Reconstruct the path by backtracking using parent pointers.
    # We use a fixed-size array for actions.
    def reconstruct_path(_):
        path_actions = -jnp.ones((max_states,), dtype=jnp.int32)
        count = jnp.array(0, dtype=jnp.int32)
        cur_state = found_state  # start at the goal state

        def cond_rec(rec_state):
            cur_state, count, path_actions = rec_state
            r, c, o = cur_state[0], cur_state[1], cur_state[2]
            return parent[r, c, o] != -1

        def body_rec(rec_state):
            cur_state, count, path_actions = rec_state
            r, c, o = cur_state[0], cur_state[1], cur_state[2]
            par = parent[r, c, o]
            act = action_taken[r, c, o]
            path_actions = path_actions.at[count].set(act)
            par_r = par // (W * 4)
            par_rem = par % (W * 4)
            par_c = par_rem // 4
            par_o = par_rem % 4
            new_state = jnp.array([par_r, par_c, par_o], dtype=jnp.int32)
            return (new_state, count + 1, path_actions)

        cur_state_final, count_final, path_actions_final = lax.while_loop(cond_rec, body_rec,
                                                                          (cur_state, count, path_actions))
        # Instead of slicing dynamically, we return the full fixed-length array and count.
        return path_actions_final, count_final

    policy_fixed, count = lax.cond(found, reconstruct_path,
                                   lambda _: (jnp.zeros((max_states,), dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
                                   operand=None)
    return policy_fixed, count


# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Create a grid (0: free, 1: wall)
    grid_np = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0]
    ], dtype=np.int32)
    grid = jax.device_put(grid_np)

    start = (0, 5)  # starting position (row, col)
    start_orientation = 1  # facing up
    goal = (3, 0)  # goal position

    policy_fixed, count = jax_shortest_path_policy(grid, start, start_orientation, goal)
    # Outside the jitted function, slice the fixed array using the count.
    policy = policy_fixed[:count]
    print("Shortest path policy (actions):", policy)
    print()

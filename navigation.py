"""
Symbolic navigator.
The field V[cell] is the cheapest cost to reach the nearest goal from that cell. The policy walks downhill on V. No learning, no enemy avoidance.
"""
import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxatari

LARGE_COST = 1e6                      # a large finite cost (not inf: inf breaks min/where)
RELAX_IT = 200
MAX_STEP = 3000
DIR_TO_ACTION = 2                     # direction 0..3 -> action 2..5


def pos_to_grid(pos: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Pixel position to maze cell index. Mirrors the env's own snap."""
    return (pos[0] + 5) // 4, (pos[1] + 3) // 4


def get_walkable(maze: chex.Array) -> chex.Array:
    """Cells belonging to the graph: at least one open direction.

    maze    -- (40, 44, 4) bool
    returns -- (40, 44) bool
    """
    return maze.any(axis=-1)


@jax.jit
def pellet_cell_indices(pellets: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Where each pellet-grid entry lands on the maze cell grid. Pellets are eaten at a specific pixel; invert that to a goal cell.

    returns -- (gx, gy), each same shape as `pellets`
    """
    n_px, n_py = jnp.shape(pellets)
    px = jnp.arange(n_px)[:, None]
    py = jnp.arange(n_py)[None, :]

    x = jnp.where(8 * px + 5 < 75, 8 * px + 5, 8 * px + 9)   # pixel-specific eat column
    y = 12 * py + 6

    gx = jnp.broadcast_to((x + 5) // 4, jnp.shape(pellets))
    gy = jnp.broadcast_to((y + 3) // 4, jnp.shape(pellets))
    return gx, gy


def pellet_goals(pellets: chex.Array, walkable: chex.Array,
                 gx: chex.Array, gy: chex.Array) -> chex.Array:
    """Scatter remaining pellets onto the maze cell grid as a goal mask.

    pellets  -- (18, 14) uint8/bool
    walkable -- (40, 44) bool
    returns  -- (40, 44) bool
    """
    mask = jnp.zeros(walkable.shape, dtype=jnp.bool_)
    mask = mask.at[gx, gy].set(pellets.astype(jnp.bool_))
    return mask & walkable          # drop any that snapped onto a wall


@jax.jit
def value_iteration(maze: chex.Array, goal_mask: chex.Array,
                    walkable: chex.Array) -> chex.Array:
    """V[cell] = cheapest cost to reach the nearest goal cell."""
    V0 = jnp.where(goal_mask, 0.0, LARGE_COST)

    def relax(V: chex.Array, _):
        step_cost = 1.0                 
        G = step_cost + V
        up = jnp.roll(G, 1, axis=1)      # neighbour at y-1
        right = jnp.roll(G, -1, axis=0)  # neighbour at x+1
        left = jnp.roll(G, 1, axis=0)    # neighbour at x-1
        down = jnp.roll(G, -1, axis=1)   # neighbour at y+1

        neighbour = jnp.stack([up, right, left, down], axis=-1)
        candidates = jnp.where(maze, neighbour, LARGE_COST)
        V = jnp.min(candidates, axis=-1)

        V = jnp.where(goal_mask, 0.0, V)
        V = jnp.where(walkable, V, LARGE_COST)
        V = jnp.minimum(V, LARGE_COST)
        return V, None

    V, _ = jax.lax.scan(relax, V0, xs=None, length=RELAX_IT)
    return V


@jax.jit
def greedy_action(V: chex.Array, maze: chex.Array, goal_mask: chex.Array,
                  pos: chex.Array, prev_dir: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Descend the field: step to the legal neighbour with the lowest value.

    returns -- (action, direction)
    """
    gx, gy = pos_to_grid(pos)
    neighbour = jnp.array([
        V[gx, gy - 1],   # up
        V[gx + 1, gy],   # right
        V[gx - 1, gy],   # left
        V[gx, gy + 1],   # down
    ])

    legal = maze[gx, gy]
    V_nbr = jnp.where(legal, neighbour, LARGE_COST)

    tie = jnp.zeros(4).at[prev_dir].add(-1e-3)   # break exact ties toward current heading
    d = jnp.argmin(V_nbr + tie).astype(jnp.int32)
    d = jnp.where(legal.any(), d, jnp.argmax(legal).astype(jnp.int32))

    # planner is cell-level but eating is a pixel test so coast on a goal cell until pellet is eaten
    d = jnp.where(goal_mask[gx, gy], prev_dir, d)
    return d + DIR_TO_ACTION, d

#--Eval / rollout--
def make_act_fn(env, maze_id: int = 0) -> Tuple[Callable, chex.Array]:
    """Build (act_fn, init_carry) for utility.rollout. 
    """
    maze = env.consts.DOF_MAZES[maze_id]
    walkable = get_walkable(maze)

    @jax.jit
    def step(obs_pellets, obs_pos, prev_dir):
        gx, gy = pellet_cell_indices(obs_pellets)
        goals = pellet_goals(obs_pellets, walkable, gx, gy)
        V = value_iteration(maze, goals, walkable)
        return greedy_action(V, maze, goals, obs_pos, prev_dir)

    def act_fn(obs, prev_dir):
        action, d = step(obs.pellets, obs.player_position, prev_dir)
        return action, d

    return act_fn, jnp.int32(0)

#--Main Loop--
def main():
    from utility import rollout, save_gif

    env = jaxatari.make("mspacman")
    act_fn, init_carry = make_act_fn(env, maze_id=0)
    out = rollout(env, act_fn, init_carry, seed=0, max_steps=MAX_STEP, render=True)

    print(f"[navigation] score={out['score']}  frames={out['steps']}")
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_gif(out["frames"], os.path.join(out_dir, "navigation.gif"))

if __name__ == "__main__":
    main()
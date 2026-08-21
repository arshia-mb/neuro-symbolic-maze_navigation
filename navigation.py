import jax 
import jax.numpy as jnp
from typing import Callable, NamedTuple
import chex

LARGE_COST = 1e6
RELAX_IT = 128
DIR_TO_ACTION = 2 #direction to action


# ----- Navigation -----
def plan(maze, goal_mask, walkable, danger):
    """Value iteration maze solver. Move towards the lowest value.
    """
    V_init = jnp.where(goal_mask, 0.0, LARGE_COST)

    @jax.checkpoint
    def relax(V, _):
        step_cost = 1.0 + danger 
        G = step_cost + V 
        up = jnp.roll(G, 1, axis=1)
        right = jnp.roll(G, -1, axis=0)
        left = jnp.roll(G, 1, axis=0)
        down = jnp.roll(G, -1, axis=1)
        neighbor = jnp.stack([up,right,left,down], axis=-1)
        V = jnp.min(jnp.where(maze, neighbor, LARGE_COST), axis=-1)
        V = jnp.where(goal_mask, 0.0, V)
        V = jnp.where(walkable, V, LARGE_COST)
        V = jnp.minimum(V, LARGE_COST)
        return V, None
    len_it = RELAX_IT
    V, _ = jax.lax.scan(relax, V_init, xs=None, length=len_it)
    return V

class GameEncoder(NamedTuple):
    # --- static data (computed once, per game) ---
    maze: chex.Array        # (H, W, 4) DOF legality
    walkable: chex.Array    # (H, W) bool
    gx: chex.Array          # goal-cell x indices (precomputed)
    gy: chex.Array          # goal-cell y indices

    # --- game-specific functions ---
    snap: Callable          # pos -> (gx, gy)
    goal: Callable          # (obs, walkable, gx, gy) -> (H,W) goal mask
    features: Callable      # (obs, maze, walkable) -> (H,W,C) net input
    enemy_pos: Callable     # obs -> enemy pixel positions (for danger later)
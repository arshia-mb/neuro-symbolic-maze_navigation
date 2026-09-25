import jax 
import jax.numpy as jnp

LARGE_COST = 1e6
RELAX_IT = 128
DIR_TO_ACTION = 2 #direction to action
GOAL_TAU = 8.0    #a goal is not a goal if danger value is more than this
# ----- Navigation -----
def plan(maze, goal_mask, walkable, cost=None, relax_it=RELAX_IT):
    """Value iteration maze solver. Distance to nearest goal cell.
    """    
    step_cost = 1.0 if cost is None else 1.0 + cost
    V_init = jnp.where(goal_mask, 0.0, LARGE_COST)
 
    def relax(V, _):
        G = step_cost + V                          # cost of entering a cell, then going on from it
        up = jnp.roll(G, 1, axis=1)
        right = jnp.roll(G, -1, axis=0)
        left = jnp.roll(G, 1, axis=0)
        down = jnp.roll(G, -1, axis=1)
        neighbor = jnp.stack([up, right, left, down], axis=-1)
        V = jnp.min(jnp.where(maze, neighbor, LARGE_COST), axis=-1)
        V = jnp.where(goal_mask, 0.0, V)
        V = jnp.where(walkable, V, LARGE_COST)
        return jnp.minimum(V, LARGE_COST), None
 
    relax_it = max(int(relax_it), walkable.shape[0] + walkable.shape[1] + 4)
    V, _ = jax.lax.scan(relax, V_init, xs=None, length=relax_it)
    return V


def greedy_action(maze, walkable, goals, cost, pos, prev_dir, snap, goal_tau=GOAL_TAU):
    """Descend navigation + LAMBDA*danger. Move to the closest pellet with the least danger.
    """
    safe = goals & (cost < goal_tau)
    goals = jnp.where(jnp.any(safe), safe, goals)
    V = plan(maze, goals, walkable, cost=cost, relax_it=2 * RELAX_IT)   # detours need more sweeps
 
    W, H = walkable.shape
    gx, gy = snap(pos)
    gx, gy = gx % W, gy % H
    nbrs = [(gx, (gy - 1) % H), ((gx + 1) % W, gy), ((gx - 1) % W, gy), (gx, (gy + 1) % H)]
    score = jnp.array([cost[x, y] + V[x, y] for x, y in nbrs])     # wraps like plan's roll (tunnels)
    score = jnp.where(maze[gx, gy], score, LARGE_COST)
 
    tie = jnp.zeros(4).at[prev_dir].add(-1e-3)                      # prefer keeping direction on ties
    d = jnp.argmin(score + tie).astype(jnp.int32)
    d = jnp.where(goals[gx, gy], prev_dir, d).astype(jnp.int32)     # coast across (safe) goal cells
    return d + DIR_TO_ACTION, d, V, goals

#--- Maze Building ---
def build_legal_moves(walkable):
    """
    Build the (W, H, 4) legality mask from a plain walkable grid.
    A move is legal if the destination cell is in-bounds AND walkable.
    Direction order matches plan()'s convention: [UP, RIGHT, LEFT, DOWN].
    """
    up_ok    = jnp.pad(walkable, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    right_ok = jnp.pad(walkable, ((0, 1), (0, 0)), constant_values=False)[1:, :]
    left_ok  = jnp.pad(walkable, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    down_ok  = jnp.pad(walkable, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

    return jnp.stack([up_ok, right_ok, left_ok, down_ok], axis=-1)
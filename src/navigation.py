import jax 
import jax.numpy as jnp

LARGE_COST = 1e6
RELAX_IT = 128
DIR_TO_ACTION = 2 #direction to action
LAMBDA = 1.0 

# ----- Navigation -----
def plan(maze, goal_mask, walkable, relax_it=RELAX_IT):
    """Value iteration maze solver. Distance to nearest goal cell.
    """    
    V_init = jnp.where(goal_mask, 0.0, LARGE_COST)

    def relax(V, _):
        step_cost = 1.0 
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

    relax_it = max(int(relax_it), walkable.shape[0] + walkable.shape[1] + 4)    
    V, _ = jax.lax.scan(relax, V_init, xs=None, length=relax_it)
    return V


def greedy_action(V, danger, maze, goal_mask, pos, prev_dir, snap, lam=LAMBDA):
    """Descend navigation + LAMBDA*danger. Move to the closest pellet with the least danger.
    """
    gx, gy = snap(pos)
    nav = jnp.array([V[gx, gy-1], V[gx+1, gy], V[gx-1, gy], V[gx, gy+1]]) 
    dng = jnp.array([danger[gx, gy-1], danger[gx+1, gy], danger[gx-1, gy], danger[gx, gy+1]]) 
    score = nav + lam * dng                    
    legal = maze[gx, gy]
    score = jnp.where(legal, score, LARGE_COST)

    tie = jnp.zeros(4).at[prev_dir].add(-1e-3)
    d = jnp.argmin(score + tie).astype(jnp.int32)
    d = jnp.where(goal_mask[gx, gy], prev_dir, d).astype(jnp.int32) #coasting until we reach the pellet pixel
    return d + DIR_TO_ACTION, d

#--- Maze Building ---
def build_legal_moves(walkable):
    """
    Build the (W, H, 4) legality mask from a plain walkable grid.
    A move is legal if the destination cell is in-bounds AND walkable.
    Direction order matches plan()'s convention: [UP, RIGHT, LEFT, DOWN].
    """
    W, H = walkable.shape
    up_ok    = jnp.pad(walkable, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    right_ok = jnp.pad(walkable, ((0, 1), (0, 0)), constant_values=False)[1:, :]
    left_ok  = jnp.pad(walkable, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    down_ok  = jnp.pad(walkable, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

    return jnp.stack([up_ok, right_ok, left_ok, down_ok], axis=-1)
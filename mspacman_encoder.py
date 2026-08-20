import jax 
import jaxatari
import jax.numpy as jnp
from navigation import plan, GameEncoder
import chex

DIR_TO_ACTION = 2
DELTA = jnp.array([[0, -1], [1, 0], [-1, 0], [0, 1]])   # (up,right,left,down)->(dx,dy)

def pos_to_grid(pos):
    """Snap function that maps grid position to pixel position.
    """
    return (pos[0] + 5) // 4, (pos[1] + 3) // 4

@jax.jit
def pellet_indices(pellets: chex.Array):
    x_n, y_n = jnp.shape(pellets)
    px = jnp.arange(x_n)[:, None]
    py = jnp.arange(y_n)[None, :]
    x = jnp.where(8 * px + 5 < 75, 8 * px + 5, 8 * px + 9)
    y = 12 * py + 6
    gx = jnp.broadcast_to((x + 5) // 4, jnp.shape(pellets))
    gy = jnp.broadcast_to((y + 3) // 4, jnp.shape(pellets))
    return gx, gy

@jax.jit
def get_goals(obs: chex.Array, walkable: chex.Array, gx: chex.Array, gy: chex.Array):
    mask = jnp.zeros(walkable.shape, dtype=jnp.bool_)
    mask = mask.at[gx, gy].set(obs.pellets.astype(jnp.bool_))
    return mask & walkable


def extract_features(obs, maze, walkable):
    """Per-cell features for the danger net. Returns (40, 44, 8)."""
    ax, ay = pos_to_grid(obs.player_position)
    agent_mask = jnp.zeros(walkable.shape, bool).at[ax, ay].set(True)
    zero_danger = jnp.zeros(walkable.shape, jnp.float32)
    dist = plan(maze, agent_mask, walkable, zero_danger)
    dist = jnp.where(walkable, dist, -1.0)

    gx = (obs.ghost_positions[:, 0] + 5) // 4
    gy = (obs.ghost_positions[:, 1] + 3) // 4
    occ = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(1.0)

    dirs = obs.ghost_actions - DIR_TO_ACTION
    vecs = DELTA[dirs].astype(jnp.float32)
    vx = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(vecs[:, 0])
    vy = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(vecs[:, 1])

    scalar = jnp.stack([dist.astype(jnp.float32), occ, vx, vy], axis=-1)
    return jnp.concatenate([maze.astype(jnp.float32), scalar], axis=-1)

def make_mspacman_encoder(env: chex.Array, maze_id=0) -> GameEncoder:
    """Create game encoder interface for agent to use
    """
    maze = env.consts.DOF_MAZES[maze_id]
    walkable = maze.any(axis=-1)

    # precompute goal indices once (needs a sample obs for the pellet shape)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    gx, gy = pellet_indices(obs.pellets)      # your existing function

    return GameEncoder(
        maze=maze,
        walkable=walkable,
        gx=gx,
        gy=gy,
        snap=pos_to_grid,                     # your existing functions, passed by name
        goal=get_goals,
        features=extract_features,
        enemy_pos=lambda obs: obs.ghost_positions,
    )
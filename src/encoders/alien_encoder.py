"""Game encoder for JAXAtari's Alien environment."""

import jax
import jax.numpy as jnp
from jax import lax

from encoders.game_encoder import GameEncoder
from navigation import build_legal_moves, plan

TILE = 4
PLAYER_WIDTH = 8
PLAYER_HEIGHT = 13
LARGE_COST = 1e6


def _walkable_from_collision(raw, tile=TILE, box=(PLAYER_WIDTH, PLAYER_HEIGHT)):
    """Return grid cells containing at least one collision-free player anchor."""
    blocked = lax.reduce_window(
        raw.astype(jnp.float32),
        -jnp.inf,
        lax.max,
        window_dimensions=box,
        window_strides=(1, 1),
        padding="VALID",
    ).astype(jnp.bool_)

    width, height = raw.shape[0] // tile, raw.shape[1] // tile
    pad_width = max(width * tile - blocked.shape[0], 0)
    pad_height = max(height * tile - blocked.shape[1], 0)
    blocked = jnp.pad(
        blocked,
        ((0, pad_width), (0, pad_height)),
        constant_values=True,
    )[: width * tile, : height * tile]
    return ~jnp.all(blocked.reshape(width, tile, height, tile), axis=(1, 3))


def _snap(pos, walkable_shape):
    gx = jnp.clip(
        (pos[0] // TILE).astype(jnp.int32), 0, walkable_shape[0] - 1
    )
    gy = jnp.clip(
        (pos[1] // TILE).astype(jnp.int32), 0, walkable_shape[1] - 1
    )
    return gx, gy


def pellet_indices(eggs):
    """Map Alien's ``(x, y, active, color)`` egg records to grid indices."""
    positions = eggs[..., :2]
    return (
        (positions[..., 0] // TILE).astype(jnp.int32),
        (positions[..., 1] // TILE).astype(jnp.int32),
    )


def _make_goal(egg_mask):
    def goal(obs, walkable, gx_grid, gy_grid):
        del obs, gx_grid, gy_grid
        return egg_mask & walkable

    return goal


def _enemy_direction(orientation):
    """Convert JAXAtariAction orientation values to (dx, dy)."""
    dx = jnp.where(
        orientation == 3,
        1.0,
        jnp.where(orientation == 4, -1.0, 0.0),
    )
    dy = jnp.where(
        orientation == 2,
        -1.0,
        jnp.where(orientation == 5, 1.0, 0.0),
    )
    return dx, dy


def _enemy_pos(obs):
    return jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1)


def _features(state, obs, maze, walkable):
    width, height = walkable.shape
    px, py = _snap(jnp.array([obs.player.x, obs.player.y]), (width, height))
    player_mask = jnp.zeros(walkable.shape, bool).at[px, py].set(True)
    player_distance = plan(maze, player_mask, walkable)
    player_distance = jnp.where(
        (player_distance < LARGE_COST) & walkable, player_distance, -1.0
    )

    enemy_pos = _enemy_pos(obs)
    ex, ey = jax.vmap(lambda pos: _snap(pos, (width, height)))(enemy_pos)
    dangerous = (obs.enemies.active > 0) & (obs.enemies_killable <= 0)
    enemy_mask = (
        jnp.zeros(walkable.shape, jnp.int32)
        .at[ex, ey]
        .max(dangerous.astype(jnp.int32))
        > 0
    )
    enemy_distance = plan(maze, enemy_mask, walkable)
    enemy_distance = jnp.where(
        (enemy_distance < LARGE_COST) & walkable, enemy_distance, -1.0
    )

    vx_values, vy_values = _enemy_direction(obs.enemies.orientation)
    vx = jnp.zeros(walkable.shape, jnp.float32).at[ex, ey].set(vx_values)
    vy = jnp.zeros(walkable.shape, jnp.float32).at[ex, ey].set(vy_values)

    scalar = jnp.stack(
        [player_distance.astype(jnp.float32), enemy_distance, vx, vy], axis=-1
    )
    return jnp.concatenate([maze.astype(jnp.float32), scalar], axis=-1)


def make_alien_encoder(env, maze_id=None, seed=0):
    """Build an 8-channel encoder for the primary Alien maze."""
    del maze_id
    obs, state = env.reset(jax.random.PRNGKey(seed))
    raw_collision = state.level.collision_map
    walkable = _walkable_from_collision(raw_collision)
    width, height = walkable.shape

    maze = build_legal_moves(walkable)
    egg_x, egg_y = pellet_indices(state.eggs)
    egg_active = state.eggs[..., 2] > 0
    egg_mask = (
        jnp.zeros(walkable.shape, jnp.int32)
        .at[egg_x, egg_y]
        .max(egg_active.astype(jnp.int32))
        > 0
    )
    gx_grid, gy_grid = jnp.meshgrid(
        jnp.arange(width), jnp.arange(height), indexing="ij"
    )

    return GameEncoder(
        maze=maze,
        walkable=walkable,
        gx=gx_grid,
        gy=gy_grid,
        snap=lambda pos: _snap(pos, (width, height)),
        goal=_make_goal(egg_mask),
        features=_features,
        enemy_pos=_enemy_pos,
        player_pos=lambda obs: jnp.stack([obs.player.x, obs.player.y]),
        score=lambda state: state.level.score[0],
        enemy_active=lambda obs, state: (
            (obs.enemies.active > 0) & (obs.enemies_killable <= 0)
        ),
        valid=lambda state: state.level.bonus_flag == 0,
    )

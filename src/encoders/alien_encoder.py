"""JAXAtari Alien adapter for the shared navigation and danger-net code."""

import jax
import jax.numpy as jnp
from jax import lax

from encoders.game_encoder import GameEncoder
from navigation import build_legal_moves, plan


TILE = 4
LARGE_COST = 1e6
PLAYER_FOOTPRINT = (8, 13)


def _walkable_from_collision(raw, footprint=PLAYER_FOOTPRINT):
    """Convert Alien's pixel collision map to a player-sized tile grid."""
    blocked = lax.reduce_window(
        raw.astype(jnp.float32),
        -jnp.inf,
        lax.max,
        window_dimensions=footprint,
        window_strides=(1, 1),
        padding="VALID",
    ) > 0

    width = (raw.shape[0] + TILE - 1) // TILE
    height = (raw.shape[1] + TILE - 1) // TILE
    padded_width = width * TILE - blocked.shape[0]
    padded_height = height * TILE - blocked.shape[1]
    blocked = jnp.pad(
        blocked,
        ((0, padded_width), (0, padded_height)),
        constant_values=True,
    )
    tiles = blocked.reshape(width, TILE, height, TILE)
    return ~jnp.all(tiles, axis=(1, 3))


def _snap(pos, shape):
    x = jnp.clip(pos[0] // TILE, 0, shape[0] - 1)
    y = jnp.clip(pos[1] // TILE, 0, shape[1] - 1)
    return x.astype(jnp.int32), y.astype(jnp.int32)


def _egg_indices(eggs, shape):
    x = jnp.clip((eggs[:, :, 0] // TILE).astype(jnp.int32), 0, shape[0] - 1)
    y = jnp.clip((eggs[:, :, 1] // TILE).astype(jnp.int32), 0, shape[1] - 1)
    return x, y


def _set_positions(mask, x, y, active):
    return mask.at[x, y].set(active | mask[x, y])


def _goal(obs, walkable, gx, gy, state):
    goals = jnp.zeros(walkable.shape, dtype=jnp.bool_)
    active_eggs = state.eggs[:, :, 2] > 0
    goals = _set_positions(goals, gx, gy, active_eggs)

    # During the bonus stage, eggs are gone and the score item is the target.
    item_x, item_y = _snap(obs.score_item_position, walkable.shape)
    item_active = state.level.bonus_flag.astype(jnp.bool_)
    goals = goals.at[item_x, item_y].set(item_active | goals[item_x, item_y])
    return goals & walkable


def _direction_delta(direction):
    direction = direction.astype(jnp.int32)
    dx = jnp.where(direction == 3, 1.0, jnp.where(direction == 4, -1.0, 0.0))
    dy = jnp.where(direction == 5, 1.0, jnp.where(direction == 2, -1.0, 0.0))
    return dx, dy


def _features(state, obs, maze, walkable):
    width, height = walkable.shape
    px, py = _snap(jnp.array([obs.player.x, obs.player.y]), walkable.shape)
    player_mask = jnp.zeros(walkable.shape, bool).at[px, py].set(True)
    player_dist = plan(maze, player_mask, walkable)
    player_dist = jnp.where(
        (player_dist < LARGE_COST) & walkable, player_dist, -1.0
    )

    enemy_x, enemy_y = _snap(
        jnp.stack([obs.enemies.x, obs.enemies.y], axis=0), walkable.shape
    )
    active = (obs.enemies.active > 0) & ~obs.enemies_killable.astype(jnp.bool_)
    enemy_mask = jnp.zeros(walkable.shape, bool).at[enemy_x, enemy_y].set(active)
    enemy_dist = plan(maze, enemy_mask, walkable)
    enemy_dist = jnp.where(
        (enemy_dist < LARGE_COST) & walkable, enemy_dist, -1.0
    )

    dx, dy = _direction_delta(obs.enemies.orientation)
    vx = jnp.zeros(walkable.shape, jnp.float32).at[enemy_x, enemy_y].set(dx * active)
    vy = jnp.zeros(walkable.shape, jnp.float32).at[enemy_x, enemy_y].set(dy * active)
    scalar = jnp.stack(
        [player_dist.astype(jnp.float32), enemy_dist, vx, vy], axis=-1
    )
    return jnp.concatenate([maze.astype(jnp.float32), scalar], axis=-1)


def _enemy_pos(obs):
    return jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1)


def make_alien_encoder(env) -> GameEncoder:
    """Build an encoder from Alien's primary-stage collision map."""
    obs, state = env.reset(jax.random.PRNGKey(0))
    walkable = _walkable_from_collision(state.level.collision_map)
    maze = build_legal_moves(walkable)
    gx, gy = _egg_indices(state.eggs, walkable.shape)

    return GameEncoder(
        maze=maze,
        walkable=walkable,
        gx=gx,
        gy=gy,
        snap=lambda pos: _snap(pos, walkable.shape),
        goal=_goal,
        features=_features,
        enemy_pos=_enemy_pos,
    )

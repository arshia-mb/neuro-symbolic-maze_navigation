"""
game_encoder for Bank Heist, matching the interface agent_mf.py's train()
requires AND matching mspacman_encoder.py's 8-channel feature schema
EXACTLY, so a DangerNet trained on one game is at least shape-compatible
with the other, and -- where the underlying game semantics genuinely
correspond -- channel-meaning-compatible too.

8-CHANNEL SCHEMA (matching mspacman_encoder.extract_features exactly):
  ch 0-3: maze legal-move mask (DOF), as float32 -- direct analog of
          Ms. Pac-Man's env.consts.DOF_MAZES; here derived FROM walkable
          via navigation.build_legal_moves (the reverse derivation
          direction vs. Ms. Pac-Man, since Bank Heist has no precomputed
          DOF constant -- functionally equivalent output.)
  ch 4:   distance-from-player-to-every-cell, via plan(), -1.0 sentinel
          for unreachable/non-walkable cells (NOT 1e6 -- matches
          mspacman_encoder's exact sentinel choice).
  ch 5:   distance-from-nearest-DANGEROUS-enemy-to-every-cell, via plan(),
          same -1.0 sentinel. "Dangerous" = obs.enemies.active > 0 -- Bank
          Heist's real ObjectObservation has no mode/frightened field like
          Ms. Pac-Man's ghosts.modes, so all active police are treated as
          dangerous. This is a genuine semantic match (Bank Heist doesn't
          appear to have a vulnerable-police mechanic), not a fallback.
  ch 6-7: enemy movement direction (vx, vy), scattered at each active
          enemy's grid cell. GENUINE ASSUMPTION, UNVERIFIED: derived from
          obs.enemies.orientation via (cos, sin), since Bank Heist's
          observation has no obs.enemies.actions field like Ms. Pac-Man's
          obs.ghost_actions. The actual unit/encoding of `orientation`
          (radians? degrees? discrete direction code?) has NOT been
          confirmed against a real observation -- verify this on your
          first real run before trusting these two channels.
"""
import jax
import jax.numpy as jnp
from jax import lax

from navigation import build_legal_moves, plan
from encoders.game_encoder import GameEncoder

TILE = 4
COLLISION_BOX = 8          # car footprint, from jax_bankheist.py consts
WALL_VALUE = 255           # check_background_collision uses `collision >= 255`
LARGE_COST = 1e6


def _walkable_from_collision(raw, tile=TILE, box=COLLISION_BOX):
    """Cell is walkable iff SOME box-sized anchor inside that cell is collision-free.

    Mirrors check_background_collision: the car occupies a `box`x`box` region and a
    position is blocked if any pixel in that region is a wall. A tile is then open
    if at least one anchor within it admits a clear box -- using a single fixed
    anchor per tile is too strict, since BankHeist's roads are ~box wide and only
    a narrow band of anchors actually fits.
    """
    blocked = lax.reduce_window(
        raw.astype(jnp.float32), -jnp.inf, lax.max,
        window_dimensions=(box, box), window_strides=(1, 1), padding="VALID",
    ) >= float(WALL_VALUE)                       # (W_px-box+1, H_px-box+1)

    W = raw.shape[0] // tile
    H = raw.shape[1] // tile

    pad_w = max(W * tile - blocked.shape[0], 0)
    pad_h = max(H * tile - blocked.shape[1], 0)
    blocked = jnp.pad(blocked, ((0, pad_w), (0, pad_h)),
                      constant_values=True)[:W * tile, :H * tile]

    tiles = blocked.reshape(W, tile, H, tile)
    return ~jnp.all(tiles, axis=(1, 3))  

def _snap(pos, walkable_shape):
    x, y = pos[0], pos[1] - 1        # match check_background_collision's y offset
    gx = jnp.clip((x // TILE).astype(jnp.int32), 0, walkable_shape[0] - 1)
    gy = jnp.clip((y // TILE).astype(jnp.int32), 0, walkable_shape[1] - 1)
    return gx, gy

def _angle_to_delta(angle):
    """orientation (degrees) -> (dx, dy) unit vector, matching the DIR_* enum
    in jax_bankheist.py: 0=UP, 90=RIGHT, 180=DOWN, 270=LEFT (y grows downward)."""
    dx = jnp.where(angle == 90.0, 1.0, jnp.where(angle == 270.0, -1.0, 0.0))
    dy = jnp.where(angle == 180.0, 1.0, jnp.where(angle == 0.0, -1.0, 0.0))
    return dx, dy

def _goal(obs, walkable, gx_grid, gy_grid):
    """Build a (W,H) boolean goal mask from currently-active banks."""
    bank_gx = jnp.clip((obs.banks.x // TILE).astype(jnp.int32), 0, walkable.shape[0] - 1)
    bank_gy = jnp.clip((obs.banks.y // TILE).astype(jnp.int32), 0, walkable.shape[1] - 1)
    active = obs.banks.active > 0

    goals = jnp.zeros(walkable.shape, dtype=bool)
    goals = goals.at[bank_gx, bank_gy].set(active | goals[bank_gx, bank_gy])
    return goals & walkable  # matches mspacman_encoder's get_goals: mask & walkable


def _enemy_pos(obs):
    return jnp.stack([obs.enemies.x, obs.enemies.y], axis=-1)


def _features(state, obs, maze, walkable):
    """8-channel spatial tensor -- see module docstring for the exact schema."""
    W, H = walkable.shape

    # ch 0-3: DOF / legal-move mask
    dof = maze.astype(jnp.float32)

    # ch 4: distance from player to every cell, -1.0 sentinel
    px, py = _snap(jnp.array([obs.player.x, obs.player.y]), (W, H))
    agent_mask = jnp.zeros(walkable.shape, bool).at[px, py].set(True)
    dist_to_player = plan(maze, agent_mask, walkable)
    dist_to_player = jnp.where((dist_to_player < LARGE_COST) & walkable, dist_to_player, -1.0)

    # ch 5: distance to nearest DANGEROUS enemy, -1.0 sentinel
    ex = jnp.clip((obs.enemies.x // TILE).astype(jnp.int32), 0, W - 1)
    ey = jnp.clip((obs.enemies.y // TILE).astype(jnp.int32), 0, H - 1)
    dangerous = obs.enemies.active > 0  # no frightened/mode field on Bank Heist enemies
    enemy_mask = jnp.zeros(walkable.shape, bool).at[ex, ey].set(dangerous)
    dist_to_enemy = plan(maze, enemy_mask, walkable)
    dist_to_enemy = jnp.where((dist_to_enemy < LARGE_COST) & walkable, dist_to_enemy, -1.0)

    # ch 6-7: enemy movement direction, from orientation -- UNVERIFIED encoding, see docstring
    active_f = dangerous.astype(jnp.float32)
    vx_vals, vy_vals = _angle_to_delta(obs.enemies.orientation)
    vx = jnp.zeros(walkable.shape, jnp.float32).at[ex, ey].set(vx_vals)
    vy = jnp.zeros(walkable.shape, jnp.float32).at[ex, ey].set(vy_vals)

    scalar = jnp.stack([dist_to_player.astype(jnp.float32), dist_to_enemy, vx, vy], axis=-1)
    return jnp.concatenate([dof, scalar], axis=-1)  # (W, H, 8)


def make_bankheist_encoder(state):
    """
    Build the encoder from a real `state` (e.g. from env.reset()). Reads
    state.map_collision directly -- no sprite files needed beyond having
    created the env itself. Returns a GameEncoder (same interface as
    make_mspacman_encoder), with an 8-channel features() matching the
    Ms. Pac-Man schema (see module docstring for exact channel semantics
    and what's verified vs. assumed).
    """
    raw = state.map_collision                    # (160, 210), [x, y]

    walkable = _walkable_from_collision(raw)
    W, H = walkable.shape

    maze = build_legal_moves(walkable)
    gx_grid, gy_grid = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing="ij")

    return GameEncoder(
        maze=maze,
        walkable=walkable,
        gx=gx_grid,
        gy=gy_grid,
        snap=lambda pos: _snap(pos, (W, H)),
        goal=_goal,
        features=_features,
        enemy_pos=_enemy_pos,
    )

"""
game_encoder for Bank Heist, matching the interface agent_mf.py's train()
requires AND matching mspacman_encoder.py's 8-channel feature schema
EXACTLY, so a DangerNet trained on one game is at least shape-compatible
with the other, and -- where the underlying game semantics genuinely
correspond -- channel-meaning-compatible too.

Built from CONFIRMED real facts (verified against jax_bankheist.py source):
  - state.map_collision is live pytree data (a jnp.array), NOT a file
    dependency -- available directly off any real `state`.
  - map_collision is indexed [x, y] (confirmed from check_background_collision).
  - Wall/collision threshold: value >= 255 (confirmed from source).
  - COLLISION_BOX=(8,8) -- tile size chosen to match this exactly.
  - BankHeist real action indices: UP=2, RIGHT=3, LEFT=4, DOWN=5.

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

NOT yet verified against a real env (sprites blocked in this sandbox):
  - map_collision's exact pixel value range (assumed wall >= 255-ish;
    WALL_THRESHOLD=128 used defensively).
  - obs.enemies.orientation's actual encoding (see ch 6-7 note above).
"""
import jax
import jax.numpy as jnp

from .bh_nav import build_legal_moves, plan
from src.game_encoder import GameEncoder

TILE = 8  # matches COLLISION_BOX exactly -- real physics granularity
WALL_THRESHOLD = 128  # defensive; change to 255 if you confirm map is binary
LARGE_COST = 1e6


def _snap(pos, walkable_shape):
    """pos: (x, y) pixel position -> (gx, gy) tile index. Simple floor
    division -- Bank Heist's map_collision is a UNIFORM pixel grid (no
    irregular offsets like Ms. Pac-Man's pellet grid had)."""
    x, y = pos[0], pos[1]
    gx = jnp.clip((x // TILE).astype(jnp.int32), 0, walkable_shape[0] - 1)
    gy = jnp.clip((y // TILE).astype(jnp.int32), 0, walkable_shape[1] - 1)
    return gx, gy


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
    vx_vals = jnp.cos(obs.enemies.orientation) * active_f
    vy_vals = jnp.sin(obs.enemies.orientation) * active_f
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
    raw = state.map_collision  # confirmed [x, y] indexed
    W_px, H_px = raw.shape

    pad_w = (-W_px) % TILE
    pad_h = (-H_px) % TILE
    if pad_w or pad_h:
        raw = jnp.pad(raw, ((0, pad_w), (0, pad_h)), constant_values=255)  # pad as WALL

    W, H = raw.shape[0] // TILE, raw.shape[1] // TILE
    tiles = raw.reshape(W, TILE, H, TILE)
    tile_max = jnp.max(tiles, axis=(1, 3))
    walkable = tile_max < WALL_THRESHOLD

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


if __name__ == "__main__":
    # Smoke test with a SYNTHETIC map_collision (since real one needs sprites):
    # a simple room, walls on the border.
    from jaxatari.environment import ObjectObservation

    W_px, H_px = 160, 210
    fake_map = jnp.zeros((W_px, H_px), dtype=jnp.float32)
    fake_map = fake_map.at[0:8, :].set(255).at[-8:, :].set(255)
    fake_map = fake_map.at[:, 0:8].set(255).at[:, -8:].set(255)

    class FakeState:
        pass
    state = FakeState()
    state.map_collision = fake_map

    enc = make_bankheist_encoder(state)
    print("walkable shape:", enc.walkable.shape)
    print("maze shape:", enc.maze.shape)

    player = ObjectObservation.create(x=jnp.array(80.0), y=jnp.array(100.0),
                                       width=jnp.array(8.0), height=jnp.array(8.0), active=jnp.array(1))
    enemies = ObjectObservation.create(
        x=jnp.array([90.0, 0.0, 0.0]), y=jnp.array([110.0, 0.0, 0.0]),
        width=jnp.full((3,), 8.0), height=jnp.full((3,), 8.0),
        active=jnp.array([1, 0, 0]),
        orientation=jnp.array([0.5, 0.0, 0.0]),
    )
    banks = ObjectObservation.create(
        x=jnp.array([60.0, 130.0, 0.0]), y=jnp.array([90.0, 40.0, 0.0]),
        width=jnp.full((3,), 8.0), height=jnp.full((3,), 8.0), active=jnp.array([1, 1, 0]))
    dynamite = ObjectObservation.create(x=jnp.array(0.0), y=jnp.array(0.0),
                                         width=jnp.array(8.0), height=jnp.array(8.0), active=jnp.array(0))

    class FakeObs:
        pass
    obs = FakeObs()
    obs.player, obs.enemies, obs.banks, obs.dynamite = player, enemies, banks, dynamite

    feats = enc.features(state, obs, enc.maze, enc.walkable)
    print("features shape:", feats.shape, "(expect (W, H, 8))")
    print("nan check:", bool(jnp.isnan(feats).any()))

    goals = enc.goal(obs, enc.walkable, enc.gx, enc.gy)
    print("goal mask sum (expect 2, matching 2 active banks):", int(goals.sum()))

    gx, gy = enc.snap((jnp.array(80.0), jnp.array(100.0)))
    print("snap(80,100) ->", (int(gx), int(gy)))

    ep = enc.enemy_pos(obs)
    print("enemy_pos shape:", ep.shape, "(expect (3,2))")
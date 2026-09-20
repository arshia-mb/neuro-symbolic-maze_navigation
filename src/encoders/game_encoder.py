from typing import Callable, NamedTuple
import chex

# ----- Encoder -----
class GameEncoder(NamedTuple):
    # --- static data (computed once, per game) ---
    maze: chex.Array        # (H, W, 4) DOF legality
    walkable: chex.Array    # (H, W) bool
    gx: chex.Array          # goal-cell x indices (precomputed)
    gy: chex.Array          # goal-cell y indices

    # --- game-specific functions ---
    snap: Callable          # pos -> (gx, gy)
    goal: Callable          # (obs, walkable, gx, gy, state) -> (H,W) goal mask
    features: Callable      # (obs, maze, walkable) -> (H,W,C) net input
    enemy_pos: Callable     # obs -> enemy pixel positions (for danger later)
    player_pos: Callable    # obs -> (2,) player pixel position
    score: Callable         # state -> scalar game score
    enemy_active: Callable  # (obs, state) -> (n,) bool, enemies that count as dangerous
    valid: Callable         # state -> bool, False once the encoder no longer matches the game
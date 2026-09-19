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
    goal: Callable          # (obs, walkable, gx, gy) -> (H,W) goal mask
    features: Callable      # (obs, maze, walkable) -> (H,W,C) net input
    enemy_pos: Callable     # obs -> enemy pixel positions (for danger later)
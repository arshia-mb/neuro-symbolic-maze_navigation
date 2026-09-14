"""
potential Shared GameEncoder interface. only used by bankheist rn
"""
from typing import NamedTuple, Callable
import chex


class GameEncoder(NamedTuple):
    maze: chex.Array
    walkable: chex.Array
    gx: chex.Array
    gy: chex.Array
    snap: Callable
    goal: Callable
    features: Callable
    enemy_pos: Callable
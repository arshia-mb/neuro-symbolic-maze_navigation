import jax
import jax.numpy as jnp

LARGE_COST = 1e6


def plan(maze, goals, walkable, n_iters=None):
    W, H = walkable.shape
    if n_iters is None:
        n_iters = W + H + 4 

    V0 = jnp.where(goals, 0.0, LARGE_COST)

    def relax(V, _):
        V_up    = jnp.pad(V, ((0, 0), (1, 0)), constant_values=LARGE_COST)[:, :-1]
        V_right = jnp.pad(V, ((0, 1), (0, 0)), constant_values=LARGE_COST)[1:, :]
        V_left  = jnp.pad(V, ((1, 0), (0, 0)), constant_values=LARGE_COST)[:-1, :]
        V_down  = jnp.pad(V, ((0, 0), (0, 1)), constant_values=LARGE_COST)[:, 1:]

        candidates = jnp.stack([V_up, V_right, V_left, V_down], axis=-1) + 1.0
        candidates = jnp.where(maze, candidates, LARGE_COST)
        best_neighbor = jnp.min(candidates, axis=-1)

        V_new = jnp.where(goals, 0.0, jnp.minimum(V, best_neighbor))
        V_new = jnp.where(walkable, V_new, LARGE_COST)
        return V_new, None

    V_final, _ = jax.lax.scan(relax, V0, xs=None, length=n_iters)
    return V_final


def build_legal_moves(walkable):
    W, H = walkable.shape

    up_ok    = jnp.pad(walkable, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    right_ok = jnp.pad(walkable, ((0, 1), (0, 0)), constant_values=False)[1:, :]
    left_ok  = jnp.pad(walkable, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    down_ok  = jnp.pad(walkable, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

    return jnp.stack([up_ok, right_ok, left_ok, down_ok], axis=-1)


if __name__ == "__main__":
    # Smoke test: 5x5 open grid, wall border only, goal in one corner
    walkable = jnp.ones((5, 5), dtype=bool)
    walkable = walkable.at[0, :].set(False).at[-1, :].set(False)
    walkable = walkable.at[:, 0].set(False).at[:, -1].set(False)

    maze = build_legal_moves(walkable)
    goals = jnp.zeros((5, 5), dtype=bool).at[1, 1].set(True)

    V = plan(maze, goals, walkable)
    print("V field")
    print(V)
    assert V[1, 1] == 0.0
    assert V[3, 3] == 4.0  # Manhattan distance through open interior
    print("test passed")
    
    
"""
import jax
import jax.numpy as jnp

LARGE_COST = 1e6


def plan(maze, goals, walkable, n_iters=None):
    W, H = walkable.shape
    if n_iters is None:
        n_iters = W + H + 4 

    V0 = jnp.where(goals, 0.0, LARGE_COST)

    def relax(V, _):
        V_up    = jnp.pad(V, ((0, 0), (1, 0)), constant_values=LARGE_COST)[:, :-1]
        V_right = jnp.pad(V, ((0, 1), (0, 0)), constant_values=LARGE_COST)[1:, :]
        V_left  = jnp.pad(V, ((1, 0), (0, 0)), constant_values=LARGE_COST)[:-1, :]
        V_down  = jnp.pad(V, ((0, 0), (0, 1)), constant_values=LARGE_COST)[:, 1:]

        candidates = jnp.stack([V_up, V_right, V_left, V_down], axis=-1) + 1.0
        candidates = jnp.where(maze, candidates, LARGE_COST)
        best_neighbor = jnp.min(candidates, axis=-1)

        V_new = jnp.where(goals, 0.0, jnp.minimum(V, best_neighbor))
        V_new = jnp.where(walkable, V_new, LARGE_COST)
        return V_new, None

    V_final, _ = jax.lax.scan(relax, V0, xs=None, length=n_iters)
    return V_final


def build_legal_moves(walkable):
    W, H = walkable.shape

    up_ok    = jnp.pad(walkable, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    right_ok = jnp.pad(walkable, ((0, 1), (0, 0)), constant_values=False)[1:, :]
    left_ok  = jnp.pad(walkable, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    down_ok  = jnp.pad(walkable, ((0, 0), (0, 1)), constant_values=False)[:, 1:]

    return jnp.stack([up_ok, right_ok, left_ok, down_ok], axis=-1)


if __name__ == "__main__":
    # Smoke test: 5x5 open grid, wall border only, goal in one corner
    walkable = jnp.ones((5, 5), dtype=bool)
    walkable = walkable.at[0, :].set(False).at[-1, :].set(False)
    walkable = walkable.at[:, 0].set(False).at[:, -1].set(False)

    maze = build_legal_moves(walkable)
    goals = jnp.zeros((5, 5), dtype=bool).at[1, 1].set(True)

    V = plan(maze, goals, walkable)
    print("V field")
    print(V)
    assert V[1, 1] == 0.0
    assert V[3, 3] == 4.0  # Manhattan distance through open interior
    print("test passed")
"""
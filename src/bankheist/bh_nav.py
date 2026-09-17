"""
Symbolic navigation for Bank Heist -- mirrors mspacman_nav.py's interface
(plan + greedy_action), reusing the generic jittable planner from
navigation.py unchanged.
"""
import jax.numpy as jnp
import jax

LARGE_COST = 1e6

# Confirmed real Bank Heist action indices (from jax_bankheist.py's ACTION_SET):
# UP=2, RIGHT=3, LEFT=4, DOWN=5. Direction order [UP,RIGHT,LEFT,DOWN] matches
# navigation.plan()'s neighbor convention.
ACTION_FOR_DIR = jnp.array([2, 3, 4, 5], dtype=jnp.int32)

def plan(maze, goals, walkable, n_iters=None):
    W, H = walkable.shape
    if n_iters is None:
        n_iters = W + H + 4  # generous upper bound on longest shortest-path

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


if __name__ == "__main__":
    # Smoke test: 5x5 open grid, wall border only, goal in one corner
    walkable = jnp.ones((5, 5), dtype=bool)
    walkable = walkable.at[0, :].set(False).at[-1, :].set(False)
    walkable = walkable.at[:, 0].set(False).at[:, -1].set(False)

    maze = build_legal_moves(walkable)
    goals = jnp.zeros((5, 5), dtype=bool).at[1, 1].set(True)

    V = plan(maze, goals, walkable)
    print("V field (expect 0 at (1,1), LARGE_COST on walls, small ints elsewhere):")
    print(V)
    assert V[1, 1] == 0.0
    assert V[3, 3] == 4.0  # Manhattan distance through open interior
    print("Smoke test passed.")

def greedy_action(V, danger, maze, goal_mask, pos, prev_dir, snap, lam=8.0):
    """
    Deterministic (argmin) action selection for evaluation/testing --
    no stochastic sampling, no log-prob (unlike the training-time policy()
    in agent_MF.py, which this deliberately mirrors but simplifies).
    """
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
    dng_nbr = jnp.array([danger[gx, gy - 1], danger[gx + 1, gy], danger[gx - 1, gy], danger[gx, gy + 1]])
    score = V_nbr + lam * dng_nbr

    legal = maze[gx, gy]
    score = jnp.where(legal, score, LARGE_COST)

    d = jnp.argmin(score)
    d = jnp.where(goal_mask[gx, gy], prev_dir, d)
    action = ACTION_FOR_DIR[d]
    return action, d
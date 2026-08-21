"""
Neuro-symbolic agent for Ms. Pac-Man: learned danger field + symbolic planner.

Pipeline per step:
    obs -> extract_features -> DangerNet(params) -> danger
        -> plan (value iteration with step_cost = 1 + softplus(danger)) -> V
        -> stochastic policy -> action, logp -> env.step

Training: REINFORCE + mean baseline. Game reward flows back through the DIFFERENTIABLE planner into DangerNet. The planner is symbolic (not learned); only the danger cost field is learned.
"""
import os
from typing import Callable, Tuple

import chex
import optax
import jax
import jaxatari
import jax.numpy as jnp
import flax.linen as nn
from jax.flatten_util import ravel_pytree

LARGE_COST = 1e6
RELAX_IT = 200
TRAIN_MAX_STEP = 200         # capped for memory
EVAL_MAX_STEP = 3000         # full episodes for rendering/eval
EPOCHS = 100
DIR_TO_ACTION = 2
DELTA = jnp.array([[0, -1], [1, 0], [-1, 0], [0, 1]])   # (up,right,left,down)->(dx,dy)


# ----- Helpers --------
def pos_to_grid(pos: chex.Array):
    return (pos[0] + 5) // 4, (pos[1] + 3) // 4


def get_walkable(maze: chex.Array):
    return maze.any(axis=-1)


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


def get_goals(pellets: chex.Array, walkable: chex.Array, gx: chex.Array, gy: chex.Array):
    mask = jnp.zeros(walkable.shape, dtype=jnp.bool_)
    mask = mask.at[gx, gy].set(pellets.astype(jnp.bool_))
    return mask & walkable


# ----- Planner (danger-consuming) -----
@jax.jit
def plan(maze, goal_mask, walkable, danger):
    """Value iteration with step_cost = 1 + softplus(danger).
    """
    V_init = jnp.where(goal_mask, 0.0, LARGE_COST)

    def relax(V, _):
        step_cost = 1.0 + jax.nn.softplus(danger)    # danger >= 0: only ever adds cost
        G = step_cost + V
        up = jnp.roll(G, 1, axis=1)
        right = jnp.roll(G, -1, axis=0)
        left = jnp.roll(G, 1, axis=0)
        down = jnp.roll(G, -1, axis=1)
        neighbor = jnp.stack([up, right, left, down], axis=-1)
        V = jnp.min(jnp.where(maze, neighbor, LARGE_COST), axis=-1)
        V = jnp.where(goal_mask, 0.0, V)
        V = jnp.where(walkable, V, LARGE_COST)
        V = jnp.minimum(V, LARGE_COST)
        return V, None

    V, _ = jax.lax.scan(relax, V_init, xs=None, length=RELAX_IT)
    return V


# ----- DangerNet ------
class DangerNet(nn.Module):
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1, 1), padding="SAME")(x)
        return x[..., 0]                              # (40, 44)


def extract_features(player_position, ghost_positions, ghost_actions, maze, walkable):
    """Per-cell features for the danger net. Returns (40, 44, 8)."""
    ax, ay = pos_to_grid(player_position)
    agent_mask = jnp.zeros(walkable.shape, bool).at[ax, ay].set(True)
    zero_danger = jnp.zeros(walkable.shape, jnp.float32)
    dist = plan(maze, agent_mask, walkable, zero_danger)
    dist = jnp.where(walkable, dist, -1.0)

    gx = (ghost_positions[:, 0] + 5) // 4
    gy = (ghost_positions[:, 1] + 3) // 4
    occ = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(1.0)

    dirs = ghost_actions - 2
    vecs = DELTA[dirs].astype(jnp.float32)
    vx = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(vecs[:, 0])
    vy = jnp.zeros(walkable.shape, jnp.float32).at[gx, gy].set(vecs[:, 1])

    scalar = jnp.stack([dist.astype(jnp.float32), occ, vx, vy], axis=-1)
    return jnp.concatenate([maze.astype(jnp.float32), scalar], axis=-1)


def policy(V, maze, goal_mask, pos, prev_dir, key, temperature=1.0):
    """Stochastic, differentiable policy. Returns (action, logp, direction)."""
    gx, gy = pos_to_grid(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])

    legal = maze[gx, gy]
    logits = -V_nbr / temperature
    logits = jnp.where(legal, logits, -1e9)

    d = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[d]

    d = jnp.where(goal_mask[gx, gy], prev_dir, d).astype(jnp.int32)   # coast on goal cell
    return d + DIR_TO_ACTION, logp, d


# ----- Training -------
def discounted_returns(rewards, gamma=0.99):
    def step(carry, r):
        carry = r + gamma * carry
        return carry, carry
    _, G = jax.lax.scan(step, 0.0, rewards[::-1])
    return G[::-1]


def reinforce_loss(logps, rewards, gamma=0.99):
    """REINFORCE + mean baseline."""
    G = discounted_returns(rewards, gamma)
    advantage = G - G.mean()
    advantage = jax.lax.stop_gradient(advantage)     # returns are targets
    return -(logps * advantage).mean()               # minus: optimizer minimizes


def episode(env, params, net, maze, walkable, key, gx, gy):
    """One game under fixed params. Returns (logps, rewards), each (T,)."""
    reset_key, key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    logps, rewards = [], []
    prev_dir = jnp.int32(0)

    for _ in range(TRAIN_MAX_STEP):
        key, act_key = jax.random.split(key)
        features = extract_features(obs.player_position, obs.ghost_positions,
                                    obs.ghost_actions, maze, walkable)
        danger = net.apply(params, features)
        goals = get_goals(obs.pellets, walkable, gx, gy)
        V = plan(maze, goals, walkable, danger)
        action, logp, prev_dir = policy(V, maze, goals, obs.player_position, prev_dir, act_key)

        obs, state, reward, done, info = env.step(state, action)
        logps.append(logp)
        rewards.append(reward)
        if bool(done):
            break

    return jnp.stack(logps), jnp.stack(rewards)


def train(env, seed=0, epochs=EPOCHS, n_episodes=1, lr=1e-3):
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    maze = env.consts.DOF_MAZES[0]                   # v1: fixed maze 0
    walkable = get_walkable(maze)
    obs, _ = env.reset(init_key)
    gx, gy = pellet_indices(obs.pellets)

    net = DangerNet()
    dummy = extract_features(obs.player_position, obs.ghost_positions,
                             obs.ghost_actions, maze, walkable)
    params = net.init(init_key, dummy)               # params born ONCE

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def batch_loss(params, key):
        keys = jax.random.split(key, n_episodes)
        total = 0.0
        for k in keys:
            logps, rewards = episode(env, params, net, maze, walkable, k, gx, gy)
            total = total + reinforce_loss(logps, rewards)
        return total / n_episodes

    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        loss, grads = jax.value_and_grad(batch_loss)(params, sub)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        flat, _ = ravel_pytree(grads)
        gnorm = jnp.abs(flat).sum()
        print(f"epoch {epoch}  loss {float(loss):.3f}  grad_norm {float(gnorm):.4f}")

    return params


# ----- Eval / rollout seam -----
def make_act_fn(env, params, maze_id: int = 0, seed: int = 0,
                greedy: bool = True) -> Tuple[Callable, Tuple]:
    """Build (act_fn, init_carry) for utility.rollout.

    act_fn(obs, carry) -> (action, carry); carry = (prev_dir, key).
    greedy=True: deterministic argmin (eval). greedy=False: sample (like training).
    """
    maze = env.consts.DOF_MAZES[maze_id]
    walkable = get_walkable(maze)
    net = DangerNet()

    @jax.jit
    def step(pellets, pos, ghost_positions, ghost_actions, prev_dir, key):
        key, act_key = jax.random.split(key)
        features = extract_features(pos, ghost_positions, ghost_actions, maze, walkable)
        danger = net.apply(params, features)
        gx, gy = pellet_indices(pellets)
        goals = get_goals(pellets, walkable, gx, gy)
        V = plan(maze, goals, walkable, danger)

        if greedy:
            gx2, gy2 = pos_to_grid(pos)
            V_nbr = jnp.array([V[gx2, gy2 - 1], V[gx2 + 1, gy2],
                               V[gx2 - 1, gy2], V[gx2, gy2 + 1]])
            legal = maze[gx2, gy2]
            V_nbr = jnp.where(legal, V_nbr, LARGE_COST)
            tie = jnp.zeros(4).at[prev_dir].add(-1e-3)   # break exact ties toward heading
            d = jnp.argmin(V_nbr + tie).astype(jnp.int32)
            d = jnp.where(goals[gx2, gy2], prev_dir, d)
            action = d + DIR_TO_ACTION
        else:
            action, _logp, d = policy(V, maze, goals, pos, prev_dir, act_key)

        return action, d, key

    def act_fn(obs, carry):
        prev_dir, key = carry
        action, d, key = step(obs.pellets, obs.player_position,
                              obs.ghost_positions, obs.ghost_actions, prev_dir, key)
        return action, (d, key)

    return act_fn, (jnp.int32(0), jax.random.PRNGKey(seed))


# ----- Main -----
def main():
    from projects.maze_solver.test import rollout, save_gif

    env = jaxatari.make("mspacman")
    params = train(env, epochs=2, n_episodes=1)      # bring-up: watch grad_norm

    act_fn, init_carry = make_act_fn(env, params, maze_id=0, greedy=True)
    out = rollout(env, act_fn, init_carry, seed=0, max_steps=EVAL_MAX_STEP, render=True)
    print(f"[agent] score={out['score']}  frames={out['steps']}")

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_gif(out["frames"], os.path.join(out_dir, "agent_v1.gif"))


if __name__ == "__main__":
    main()
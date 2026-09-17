"""
Neuro-symbolic agent for Bank Heist: learned danger field (DangerNet) +
symbolic planner (navigation.plan). Adapted from the Ms. Pac-Man agent_mf.py
architecture -- same DangerNet, same policy fusion, same REINFORCE loop.
What changed for Bank Heist (all confirmed against real source, not assumed):

  - game_encoder: BankHeistEncoder (bankheist_grid_encoder.py) instead of
    mspacman_encoder -- reads state.map_collision directly (live pytree
    data, no separate wall-extraction step needed).
  - navigation.plan: reimplemented generically (navigation.py) since the
    original file imported a `navigation` module we didn't have; verified
    correct via a standalone shortest-path smoke test.
  - Enemies: obs.enemies.x/y/active (3 police cars) instead of
    obs.ghost_positions -- Bank Heist's real observation structure.
  - Action mapping: explicit lookup array ACTION_FOR_DIR = [2,3,4,5]
    (UP,RIGHT,LEFT,DOWN), confirmed against Bank Heist's real ACTION_SET,
    rather than the original file's implicit "+DIR_TO_ACTION" trick --
    functionally equivalent for this game (verified), but explicit avoids
    silently breaking if ever ported to a game with a different action order.
  - Goal = active banks (obs.banks.x/y/active) instead of pellets.
  - Death penalty uses state.player_lives (confirmed field on BankHeistState
    -- NOT state.lives, which doesn't exist; obs.lives is a separate field
    on BankHeistObservation. Caught via a real run's AttributeError.).

NOT yet verified end-to-end against the real env in this sandbox (sprites
blocked) -- validated here via a fake env + synthetic map_collision that
matches all confirmed real shapes. Run locally and report back the training
log, same pattern as every other real-env script in this project.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as fs
import optax
import numpy as np

from .bh_nav import plan
from grid_encoder import make_bankheist_encoder

MAX_EPISODE_LEN = 500
EPOCHS = 300
N_ENV = 8
MAX_WARMUP = 60

DANGER_MAX = 10.0
LAMBDA = 8.0

DEATH_PENALTY = 50.0
DANGER_WEIGHT = 0.05
DANGER_RADIUS = 10

# Explicit direction -> real action lookup (confirmed against BankHeist's
# real ACTION_SET: UP=2, RIGHT=3, LEFT=4, DOWN=5). Direction order [UP,
# RIGHT, LEFT, DOWN] matches navigation.plan()'s neighbor convention.
ACTION_FOR_DIR = jnp.array([2, 3, 4, 5], dtype=jnp.int32)


# ----- DangerNet (unchanged from the original file -- game-agnostic CNN) -----
class DangerNet(nn.Module):
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1, 1), padding="SAME")(x)
        return jax.nn.sigmoid(x[..., 0]) * DANGER_MAX


def policy(V, danger, maze, goal_mask, pos, prev_dir, key, snap, temperature=1.0):
    """Stochastic, differentiable policy -- same fusion formula as the
    original file: score = V + LAMBDA * danger, softmax over legal moves."""
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
    dng_nbr = jnp.array([danger[gx, gy - 1], danger[gx + 1, gy], danger[gx - 1, gy], danger[gx, gy + 1]])
    score = V_nbr + LAMBDA * dng_nbr

    legal = maze[gx, gy]
    logits = -score / temperature
    logits = jnp.where(legal, logits, -1e9)

    d = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[d]

    d = jnp.where(goal_mask[gx, gy], prev_dir, d)
    action = ACTION_FOR_DIR[d]
    return action, logp, d


def reinforce_loss(logps, rewards, dones, gamma=0.99):
    mask = ((jnp.cumsum(dones.astype(jnp.int32)) - dones.astype(jnp.int32)) == 0).astype(jnp.float32)
    denom = jnp.maximum(mask.sum(), 1.0)
    rewards = rewards * mask

    def discounted_returns(carry, r):
        carry = r + gamma * carry
        return carry, carry
    _, G = jax.lax.scan(discounted_returns, 0.0, rewards[::-1])

    G = G[::-1]
    baseline = (G * mask).sum() / denom
    advantage = jax.lax.stop_gradient(G - baseline)
    loss = -(logps * advantage * mask).sum() / denom
    total_return = (rewards * mask).sum()
    return loss, total_return


def save_params(params, path):
    with open(path, "wb") as f:
        f.write(fs.to_bytes(params))


def load_params(params_template, path):
    with open(path, "rb") as f:
        return fs.from_bytes(params_template, f.read())


def train(env, model_path=None, seed=0, epochs=EPOCHS, episode_len=MAX_EPISODE_LEN, lr=3e-4):
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    os.makedirs("outputs", exist_ok=True)

    obs, state = env.reset(init_key)
    game_encoder = make_bankheist_encoder(state)  # built once from the initial state's map

    maze, walkable = game_encoder.maze, game_encoder.walkable
    gx, gy = game_encoder.gx, game_encoder.gy
    snap, get_goal, features = game_encoder.snap, game_encoder.goal, game_encoder.features

    net = DangerNet()
    dummy = features(state, obs, maze, walkable)
    if model_path:
        template = net.init(jax.random.PRNGKey(0), dummy)
        params = load_params(template, model_path)
    else:
        params = net.init(init_key, dummy)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)

    def episode_loss(params, key):
        reset_key, noop_key, warmup_key, key = jax.random.split(key, 4)
        obs, state = env.reset(reset_key)

        n_warmup = jax.random.randint(noop_key, (), 0, MAX_WARMUP)

        def warmup_step(carry, i):
            obs, state, key = carry
            key, action_key = jax.random.split(key)
            action = jax.random.randint(action_key, (), 0, 18)  # real BankHeist action count
            obs_n, state_n, _, done_n, _ = env.step(state, action)
            warmup = (i < n_warmup) & (~done_n)
            obs = jax.tree_util.tree_map(lambda new, old: jnp.where(warmup, new, old), obs_n, obs)
            state = jax.tree_util.tree_map(lambda new, old: jnp.where(warmup, new, old), state_n, state)
            return (obs, state, key), None

        (obs, state, _), _ = jax.lax.scan(warmup_step, (obs, state, warmup_key), jnp.arange(MAX_WARMUP))

        prev_dir = jnp.int32(0)
        init_carry = (state, obs, prev_dir, state.player_lives, key)

        def step(carry, _):
            state, obs, prev_dir, prev_lives, key = carry
            key, act_key = jax.random.split(key)

            danger = net.apply(params, features(state, obs, maze, walkable))
            goals = get_goal(obs, walkable, gx, gy)
            V = jax.lax.stop_gradient(plan(maze, goals, walkable))
            # policy() expects a single (x,y) pos; obs.player has separate
            # .x/.y scalars (confirmed BankHeist ObjectObservation structure),
            # so pack them here rather than assuming a combined .position
            # field like Ms. Pac-Man's obs.player_position.
            pos = jnp.array([obs.player.x, obs.player.y])
            action, logp, prev_dir = policy(V, danger, maze, goals, pos, prev_dir, act_key, snap)

            obs, state, reward, done, _ = env.step(state, action)

            # Police-proximity penalty (analogous to the original's ghost penalty)
            pgx, pgy = snap(jnp.array([obs.player.x, obs.player.y]))
            agent_mask = jnp.zeros(walkable.shape, bool).at[pgx, pgy].set(True)
            dist = jax.lax.stop_gradient(plan(maze, agent_mask, walkable))
            enemy_gx = jnp.clip((obs.enemies.x // 8).astype(jnp.int32), 0, walkable.shape[0] - 1)
            enemy_gy = jnp.clip((obs.enemies.y // 8).astype(jnp.int32), 0, walkable.shape[1] - 1)
            nearest = jnp.min(jnp.where(obs.enemies.active > 0, dist[enemy_gx, enemy_gy], 1e6))
            prox_penalty = jnp.maximum(DANGER_RADIUS - nearest, 0.0)

            lives = state.player_lives
            died = (lives < prev_lives).astype(jnp.float32)
            reward = reward - DEATH_PENALTY * died - DANGER_WEIGHT * prox_penalty

            new_carry = (state, obs, prev_dir, lives, key)
            return new_carry, (logp, reward, done)

        final_carry, (logps, rewards, dones) = jax.lax.scan(step, init_carry, xs=None, length=episode_len)
        loss, total_return = reinforce_loss(logps, rewards, dones)
        return loss, total_return

    def batch_loss(params, key):
        keys = jax.random.split(key, N_ENV)
        losses, returns = jax.vmap(episode_loss, in_axes=(None, 0))(params, keys)
        return losses.mean(), returns.mean()

    @jax.jit
    def update(params, opt_state, key):
        (loss, total_return), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        return params, opt_state, loss, total_return, gnorm

    avg_return = None
    returns_log = []
    best_return = -1e9
    best_model = None
    fsave = True
    start = time.time()

    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss, total_return, gnorm = update(params, opt_state, sub)

        avg_return = float(total_return) if avg_return is None else 0.9 * avg_return + 0.1 * float(total_return)
        returns_log.append(float(total_return))

        if epoch % 10 == 0:
            elapsed = time.time() - start
            per_epoch = elapsed / (epoch + 1)
            eta = per_epoch * (epochs - epoch - 1)
            print(f"epoch {epoch}  return {float(total_return):.0f}  avg {avg_return:.0f}  "
                  f"gnorm {float(gnorm):.2f}  [ETA {eta/60:.1f} min]")

        if float(total_return) > best_return:
            best_return = float(total_return)
            best_model = params
            fsave = True

        if epoch % 100 == 0:
            save_params(params, f"outputs/bankheist_ckpt_epoch{epoch}.msgpack")
            if fsave and best_model is not None:
                save_params(best_model, "outputs/bankheist_best.msgpack")
                fsave = False

    params = best_model if best_model is not None else params
    save_params(params, "outputs/bankheist_params.msgpack")
    np.save("outputs/bankheist_returns.npy", np.array(returns_log))
    print(f"final return: {returns_log[-1]:.0f}  last_avgs: {np.mean(returns_log[-50:]):.0f}  "
          f"best score: {float(best_return):.2f}")
    return params


def main():
    import jaxatari
    env = jaxatari.make("bankheist")
    params = train(env, model_path=None, epochs=EPOCHS)


if __name__ == "__main__":
    main()
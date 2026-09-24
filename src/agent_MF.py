"""
Neuro-symbolic agent for solving mazes with enemy avoidance: learned danger field + symbolic planner.
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"      # use more of the 6GB (default holds back)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"    # on-demand alloc, less fragmentation

import time

import jax
import jax.numpy as jnp
import jaxatari
import flax.linen as nn
import flax.serialization as fs
import optax
import numpy as np


from navigation import plan, LARGE_COST, DIR_TO_ACTION


# ----- Hyperparameters -----
# training
MAX_EPISODE_LEN = 800  # steps per training episode
MAX_WARMUP = 60        # up to this many random steps before each episode (varied starts)
N_ENV = 8              # parallel episodes per update
EPOCHS = 300
LR = 3e-4

# danger field and decision rule
DANGER_MAX = 10.0    #danger output range is [0, DANGER_MAX]
LAMBDA = 8.0         #weight of danger cost relative to navigation distance

# reward shaping and auxiliary loss
DEATH_PENALTY = 50.0
DANGER_RADIUS = 10    # enemies closer than this (cells) are penalized
DANGER_WEIGHT = 0.05  # per-step proximity penalty weight
AUX_WEIGHT = 0.05     # weight of the danger-anchoring auxiliary loss

OUT_DIR = "outputs"

# ----- DangerNet -----
class DangerNet(nn.Module):
    """Conv net mapping encoder features (W, H, 8) to a danger field (W, H) in [0, DANGER_MAX].
 
    - Drops channel 4 (distance to the agent), so danger cannot depend on where the agent is: it describes the map relative to enemies only.
    - Gated on channel 5; if no dangerous enemy is reachable, the output is exactly zero.
    """
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        has_danger = jnp.any(x[..., 5:6] >= 0, axis=(-3, -2, -1))
        x = jnp.concatenate([x[..., :4], x[..., 5:]], axis=-1)   # drop agent-distance channel
 
        x = nn.relu(nn.Conv(self.hidden, (3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(self.hidden, (3, 3), padding="SAME")(x))
        x = nn.Conv(1, (1, 1))(x)
        out = jax.nn.sigmoid(x[..., 0]) * DANGER_MAX
 
        return jnp.where(has_danger[..., None, None], out, 0.0)

def danger_cost(danger):
    """Convex reshaping of danger into decision cost (small far away, steep up close)."""
    return danger ** 2 / DANGER_MAX


# --- Decision rule ---
def policy(V, danger, maze, goal_mask, pos, prev_dir, key, snap, temperature=1.0):
    """Stochastic, differentiable policy.
    
    On a goal cell the agent keeps its previous direction (coasting) so it finishes crossing the pellet.
    """
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
    dng_nbr = jnp.array([danger[gx, gy - 1], danger[gx + 1, gy], danger[gx - 1, gy], danger[gx, gy + 1]])
    dng_nbr = danger_cost(dng_nbr)
    score = V_nbr + LAMBDA * dng_nbr

    logits = jnp.where(maze[gx, gy], -score / temperature, -1e9)
    d = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[d]

    d = jnp.where(goal_mask[gx, gy], prev_dir, d) #coasting rule
    return d + DIR_TO_ACTION, logp, d

# --- Losses ---
def valid_mask(dones):
    """1.0 for steps up to and including the first `done`, 0.0 afterwards."""
    d = dones.astype(jnp.int32)
    return ((jnp.cumsum(d) - d) == 0).astype(jnp.float32)

def reinforce_loss(logps, rewards, dones, gamma=0.99):
    """REINFORCE with a mean-return baseline over the valid (pre-terminal) steps.
    Returns (loss, total_return)."""
    mask = valid_mask(dones)
    denom = jnp.maximum(mask.sum(), 1.0)
    rewards = rewards * mask

    def discounted(carry, r):
        carry = r + gamma * carry
        return carry, carry
    _, G = jax.lax.scan(discounted, 0.0, rewards[::-1])
    G = G[::-1]

    baseline = (G * mask).sum() / denom
    advantage = jax.lax.stop_gradient(G - baseline)
    loss = -(logps * advantage * mask).sum() / denom
    return loss, rewards.sum()

# --- Checkpoints ---
def save_params(params, path):
    with open(path, "wb") as f:
        f.write(fs.to_bytes(params))
 
 
def load_params(params_template, path):
    with open(path, "rb") as f:
        return fs.from_bytes(params_template, f.read())
    
# ----- Training -----   
def train(env, enc, init_path=None, seed=0, epochs=EPOCHS, episode_len=MAX_EPISODE_LEN, lr=LR, name="mspacman"):
    """Train the danger net with REINFORCE + auxiliary loss. Returns the best parameters.
 
    Writes to OUT_DIR: ckpt_epoch{N}.msgpack every 100 epochs, {name}_best.msgpack
    (best batch return so far), {name}_params.msgpack (final best) and {name}_returns.npy (per-epoch returns).
    `init_path` optionally resumes from a saved checkpoint.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    os.makedirs(OUT_DIR, exist_ok=True)

    maze, walkable = enc.maze, enc.walkable
    gx, gy, snap = enc.gx, enc.gy, enc.snap

    obs, state = env.reset(init_key)
    net = DangerNet()
    dummy = enc.features(state, obs, maze, walkable)
    if init_path:
        params = load_params(net.init(jax.random.PRNGKey(0), dummy), init_path)
    else:
        params = net.init(init_key, dummy)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)

    def episode_loss(params,key):
        reset_key, warm_len_key, warm_key, key = jax.random.split(key, 4)
        obs, state = env.reset(reset_key)

        #random-length warmup -> varied start states
        n_warmup = jax.random.randint(warm_len_key, (), 0, MAX_WARMUP)
        def warmup_step(carry, i):
            obs, state, key = carry
            key, action_key = jax.random.split(key)
            action = jax.random.randint(action_key, (), 0, 6)       # Atari actions 0..5
            obs_n, state_n, _, done_n, _ = env.step(state, action)
            advance = (i < n_warmup) & (~done_n)                    # stop if the agent dies
            obs = jax.tree_util.tree_map(lambda new, old: jnp.where(advance, new, old), obs_n, obs)
            state = jax.tree_util.tree_map(lambda new, old: jnp.where(advance, new, old), state_n, state)
            return (obs, state, key), None

        (obs, state, _), _ = jax.lax.scan(warmup_step, (obs, state, warm_key), jnp.arange(MAX_WARMUP))

        # episode
        def step(carry, _):
            state, obs, prev_dir, prev_lives, key = carry
            key, act_key = jax.random.split(key)

            feats = enc.features(state, obs, maze, walkable)
            danger = net.apply(params, feats)
            goals = enc.goal(obs, walkable, gx, gy)        
            V = jax.lax.stop_gradient(plan(maze, goals, walkable))
            action, logp, prev_dir = policy(V, danger, maze, goals, enc.player_pos(obs), prev_dir, act_key, snap)
           
            # auxiliary target -> proximity to dangerous enemies, nonzero within DANGER_RADIUS
            d_enemy = feats[..., 5]
            target = jnp.where(d_enemy >= 0,
                               jnp.clip(1.0 - d_enemy / DANGER_RADIUS, 0.0, 1.0),
                               0.0) * DANGER_MAX
            aux = jnp.mean((danger - jax.lax.stop_gradient(target)) ** 2)
 
            obs, state, reward, done, _ = env.step(state, action)

            # proximity penalty -> maze distance from the agent to the nearest dangerous enemy
            px, py = snap(enc.player_pos(obs))
            agent_mask = jnp.zeros(walkable.shape, bool).at[px, py].set(True)
            dist = jax.lax.stop_gradient(plan(maze, agent_mask, walkable))
            ex, ey = jax.vmap(snap)(enc.enemy_pos(obs))
            nearest = jnp.min(jnp.where(enc.enemy_active(obs, state), dist[ex, ey], LARGE_COST))
            prox_penalty = jnp.maximum(DANGER_RADIUS - nearest, 0.0)

            # Death Penalty 
            lives = state.lives # Ms. Pac-Man / Pacman field
            died = (lives < prev_lives).astype(jnp.float32)

            reward = reward - DEATH_PENALTY * died - DANGER_WEIGHT * prox_penalty

            new_carry = (state, obs, prev_dir, lives, key)
            output = (logp, reward, done, aux)
            return new_carry, output

        init_carry = (state, obs, jnp.int32(0), state.lives, key)
        final_carry, (logps, rewards, dones, auxs) = jax.lax.scan(step, init_carry, xs=None, length=episode_len)
    
        pg_loss, total_return = reinforce_loss(logps, rewards, dones)
        mask = valid_mask(dones)
        aux_loss = (auxs * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        loss = pg_loss + AUX_WEIGHT * aux_loss
        return loss, (total_return, aux_loss)

    def batch_loss(params, key):
        keys = jax.random.split(key, N_ENV)
        losses, (returns, auxs) = jax.vmap(episode_loss, in_axes=(None, 0))(params, keys)
        return losses.mean(), (returns.mean(), auxs.mean())

    @jax.jit
    def update(params, opt_state, key):
        (loss, (total_return, aux_loss)), grads = \
            jax.value_and_grad(batch_loss, has_aux=True)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        return params, opt_state, loss, total_return, aux_loss, gnorm

    # --- training loop ---
    returns_log, avg_return = [], None
    best_return, best_params, best_unsaved = -1e9, None, False
    start = time.time()

    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss, total_return, aux_loss, gnorm = update(params, opt_state, sub)

        ret = float(total_return)
        returns_log.append(ret)
        avg_return = ret if avg_return is None else 0.9 * avg_return + 0.1 * ret

        if ret > best_return:
            best_return, best_params, best_unsaved = ret, params, True

        if epoch % 10 == 0:
            elapsed = time.time() - start 
            per_epoch = elapsed / (epoch + 1)
            eta = per_epoch * (epochs - epoch - 1)
            print(f"epoch {epoch}  return {ret:.0f}  avg {avg_return:.0f} aux {float(aux_loss):.2f}  gnorm {float(gnorm):.2f}  [ETA {eta / 60:.1f} min]")

        if epoch % 100 == 0:
            save_params(params, f"{OUT_DIR}/ckpt_epoch{epoch}.msgpack")
            if best_unsaved:
                save_params(best_params, f"{OUT_DIR}/{name}_best.msgpack")
                best_unsaved = False

    params = best_params if best_params is not None else params
    save_params(params, f"{OUT_DIR}/{name}_params.msgpack")
    np.save(f"{OUT_DIR}/{name}_returns.npy", np.array(returns_log))
    print(f"final return: {returns_log[-1]:.0f}  last-50 mean: {np.mean(returns_log[-50:]):.0f} best: {best_return:.2f}")
    return params

# ----- Main -----
def main():
    from encoders.mspacman_encoder import make_mspacman_encoder
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env)
    train(env, enc)
 
if __name__ == "__main__":
    main()
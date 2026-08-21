"""
Neuro-symbolic agent for solving mazes with enemy avoidance: learned danger field + symbolic planner.
"""

import chex
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"      # use more of the 6GB (default holds back)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"    # on-demand alloc, less fragmentation
import jax
import jax.numpy as jnp
import jaxatari
import flax.linen as nn
import flax.serialization as fs
import optax
from navigation import plan
import numpy as np
#from jax.flatten_util import ravel_pytree
import time

LARGE_COST = 1e6
DIR_TO_ACTION = 2 #direction to action
MAX_EPISODE_LEN = 500
EPOCHS = 300
N_ENV = 6



# ----- DangerNet -----
class DangerNet(nn.Module):
    """CNN calculating the danger value for each node based on feature inputs.
    """
    hidden: int = 32
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden, (3,3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden, (3,3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1,1), padding="SAME")(x)
        return x[..., 0] #maze shape

def policy(V, maze, goal_mask, pos, prev_dir, key, snap, temperature=1.0):
    """Stochastic, differentiable policy.
    """
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])

    legal = maze[gx, gy]
    logits = -V_nbr / temperature
    logits = jnp.where(legal, logits, -1e9)

    d = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[d]

    d = jnp.where(goal_mask[gx, gy], prev_dir, d).astype(jnp.int32)   # coast on goal cell
    return d + DIR_TO_ACTION, logp, d


def reinforce_loss(logps, rewards, dones, gamma=0.99):
    """REINFORCE loss for valid (pre terminal steps).
    """
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

# ----- Training -----
def save_params(params, path):
    with open(path, "wb") as f:
        f.write(fs.to_bytes(params))

def load_params(params_template, path):
    with open(path, "rb") as f:
        return fs.from_bytes(params_template, f.read())
    
def train(env, game_encoder, model_path, seed=0, epochs=EPOCHS, episode_len=MAX_EPISODE_LEN, lr=1e-3):
    """Training the agent using by doing batch learning.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    os.makedirs("outputs", exist_ok=True)

    obs, _ = env.reset(init_key)

    maze     = game_encoder.maze
    walkable = game_encoder.walkable
    gx, gy   = game_encoder.gx, game_encoder.gy
    snap     = game_encoder.snap
    get_goal = game_encoder.goal
    features = game_encoder.features

    net = DangerNet()
    dummy = features(obs, maze, walkable)
    if model_path:
        template = net.init(jax.random.PRNGKey(0), dummy)
        params = load_params(template, model_path)
    else:
        params = net.init(init_key, dummy)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def episode_loss(params,key):
        reset_key, key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        prev_dir = jnp.int32(0)
        init_carry = (state, obs, prev_dir, key)

        def step(carry, _):
            """Running one step of the game with fixed length.
            """
            state, obs, prev_dir, key = carry
            key, act_key = jax.random.split(key)
            danger = net.apply(params, features(obs, maze, walkable))
            goals = get_goal(obs, walkable, gx, gy)
            V = plan(maze, goals, walkable, danger)
            action, logp, prev_dir = policy(V, maze, goals, obs.player_position, prev_dir, act_key, snap)

            obs, state, reward, done, _ = env.step(state, action)
            new_carry = (state, obs, prev_dir, key)
            output = (logp, reward, done)

            return new_carry, output

        final_carry, (logps, rewards, dones) = jax.lax.scan(step, init_carry, xs=None, length=episode_len)
        loss, total_return = reinforce_loss(logps, rewards, dones)
        return loss, total_return

    def batch_loss(params, key):
        keys = jax.random.split(key, N_ENV)
        losses, returns = jax.vmap(episode_loss, in_axes=(None, 0))(params,keys)
        return losses.mean(), returns.mean()

    @jax.jit
    def update(params, opt_state, key):
        (loss, total_return), grads = jax.value_and_grad(batch_loss, has_aux=True)(params,key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        return params, opt_state, loss, total_return, gnorm

    avg_return = None
    returns_log = []
    best_return = 0
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
            print(f"epoch {epoch}  return {float(total_return):.0f}  avg {avg_return:.0f}  gnorm {float(gnorm):.2f}  [ETA {eta/60:.1f} min]")

        if epoch % 100 == 0:
            save_params(params, f"outputs/ckpt_epoch{epoch}.msgpack")

        if float(total_return) > best_return:
            best_return = float(total_return)
            save_params(params, "outputs/mspacman_best.msgpack")

    save_params(params, "outputs/mspacman_params.msgpack")
    np.save("outputs/mspacman_returns.npy", np.array(returns_log))
    print(f"final return: {returns_log[-1]:.0f}  last_avgs: {np.mean(returns_log[-50:]):.0f}  best score: {float(best_return):.2f}")
    return params

# ----- Main -----
def main():
    from mspacman_encoder import make_mspacman_encoder
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env)
    model = None
    params = train(env, enc, model, epochs=EPOCHS, )

if __name__ == "__main__":
    main()
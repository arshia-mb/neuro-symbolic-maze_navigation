"""
Neuro-symbolic agent for solving mazes with enemy avoidance: learned danger field + symbolic planner.
"""

import chex
import jax
import jaxatari
import jax.numpy as jnp
import flax.linen as nn
import optax
from navigation import plan, GameEncoder
from jax.flatten_util import ravel_pytree

LARGE_COST = 1e6
MAX_EPISODE_LEN = 500
EPOCHS = 2
DIR_TO_ACTION = 2 #direction to action

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
def train(env: chex.Array, game_encoder, seed=0, epochs=EPOCHS, episode_len=MAX_EPISODE_LEN, lr=1e-3):
    """Training the agent using by doing batch learning.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    obs, _ = env.reset(init_key)

    maze     = game_encoder.maze
    walkable = game_encoder.walkable
    gx, gy   = game_encoder.gx, game_encoder.gy
    snap     = game_encoder.snap
    get_goal = game_encoder.goal
    features = game_encoder.features

    net = DangerNet()
    dummy = features(obs, maze, walkable)
    params = net.init(init_key, dummy)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    #do i even need this??
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

    for epoch in range(epochs):
        key, sub = jax.random.split(key)

        (loss, total_return), grads = jax.value_and_grad(episode_loss, has_aux=True)(params,sub)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        flat, _ = ravel_pytree(grads)          # flatten all param grads into one 1-D array
        gnorm = jnp.abs(flat).sum() 

        print(f"epoch {epoch}  loss {float(loss):.3f}  grad_norm {float(gnorm):.4f} total_return: {float(total_return)}")
    
    return params

# ----- Main -----
def main():
    from mspacman_encoder import make_mspacman_encoder
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env)
    params = train(env, enc, epochs=2)   # start with 2 to confirm it runs

if __name__ == "__main__":
    main()
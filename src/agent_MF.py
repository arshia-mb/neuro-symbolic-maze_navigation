"""
Neuro-symbolic agent for solving mazes with enemy avoidance: learned danger field + symbolic planner.
"""
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
import time

LARGE_COST = 1e6

DIR_TO_ACTION = 2 #direction to action

MAX_EPISODE_LEN = 800
EPOCHS = 300
N_ENV = 8
MAX_WARMUP = 60

DANGER_MAX = 10.0
LAMBDA = 8.0

DEATH_PENALTY = 50.0
DANGER_WEIGHT = 0.05
DANGER_RADIUS = 10
AUX_WEIGHT = 0.05


# ----- DangerNet -----
class DangerNet(nn.Module):
    """CNN calculating the danger value for each node based on feature inputs.
    """
    hidden: int = 32
    #dist_scale: float = 20.0

    @nn.compact
    def __call__(self, x):
        #dof, d_agent, d_enemy, vdir = x[..., :4], x[..., 4:5], x[..., 5:6], x[..., 6:8]
        #def prox(d): #normalized proximity
        #    return jnp.where(d >=0, jnp.clip(1.0 - d / self.dist_scale, 0.0, 1.0), 0.0) 
        #x = jnp.concatenate([dof, prox(d_agent), prox(d_enemy), vdir], axis=-1) #normalized channels

        has_danger = jnp.any(x[..., 5:6] >= 0, axis=(-3, -2, -1))

        x = jnp.concatenate([x[..., :4], x[..., 5:]], axis=-1)
        x = nn.Conv(self.hidden, (3,3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden, (3,3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1, 1))(x)
        out = jax.nn.sigmoid(x[..., 0]) * DANGER_MAX

        return jnp.where(has_danger[..., None, None], out, 0.0)
    
def policy(V, danger, maze, goal_mask, pos, prev_dir, key, snap, temperature=1.0):
    """Stochastic, differentiable policy.
    """
    gx, gy = snap(pos)
    V_nbr = jnp.array([V[gx, gy - 1], V[gx + 1, gy], V[gx - 1, gy], V[gx, gy + 1]])
    dng_nbr = jnp.array([danger[gx, gy - 1], danger[gx + 1, gy], danger[gx - 1, gy], danger[gx, gy + 1]])
    dng_nbr = dng_nbr ** 2 / DANGER_MAX
    score = V_nbr + LAMBDA * dng_nbr

    legal = maze[gx, gy]
    logits = -score / temperature
    logits = jnp.where(legal, logits, -1e9)

    d = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[d]

    d = jnp.where(goal_mask[gx, gy], prev_dir, d)
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
    
def train(env, game_encoder, model_path, seed=0, epochs=EPOCHS, episode_len=MAX_EPISODE_LEN, lr=3e-4):
    """Training the agent using by doing batch learning.
    """
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    os.makedirs("outputs", exist_ok=True)

    obs, state = env.reset(init_key)

    maze     = game_encoder.maze
    walkable = game_encoder.walkable
    gx, gy   = game_encoder.gx, game_encoder.gy
    snap     = game_encoder.snap
    get_goal = game_encoder.goal
    features = game_encoder.features

    net = DangerNet()
    dummy = features(state, obs, maze, walkable)
    if model_path:
        template = net.init(jax.random.PRNGKey(0), dummy)
        params = load_params(template, model_path)
    else:
        params = net.init(init_key, dummy)
    optimizer = optax.chain( optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)

    def episode_loss(params,key):
        reset_key, noop_key, warmup_key, key = jax.random.split(key, 4)
        obs, state = env.reset(reset_key)

        n_warmup = jax.random.randint(noop_key, (), 0, MAX_WARMUP)
        #Random Start
        def warmup_step(carry, i):
            obs, state, key = carry
            key, action_key = jax.random.split(key)
            action = jax.random.randint(action_key, (), 0, 6) #random-atari action
            obs_n, state_n, _, done_n, _ = env.step(state, action)

            warmup = (i < n_warmup) & (~done_n)
            obs   = jax.tree_util.tree_map(lambda new, old: jnp.where(warmup, new, old), obs_n, obs)
            state = jax.tree_util.tree_map(lambda new, old: jnp.where(warmup, new, old), state_n, state)
            return (obs, state, key), None

        (obs, state, _), _ = jax.lax.scan(warmup_step, (obs, state, warmup_key), jnp.arange(MAX_WARMUP))

        prev_dir = jnp.int32(0)
        init_carry = (state, obs, prev_dir, state.lives, key)

        def step(carry, _):
            """Running one step of the game with fixed length.
            """
            state, obs, prev_dir, prev_lives, key = carry
            key, act_key = jax.random.split(key)

            feats = features(state, obs, maze, walkable)

            danger = net.apply(params, feats)
            d_ghosts = feats[..., 5]            

            goals = get_goal(obs, walkable, gx, gy)
            V = jax.lax.stop_gradient(plan(maze, goals, walkable))  
            action, logp, prev_dir = policy(V, danger, maze, goals, obs.player_position, prev_dir, act_key, snap)
           
            obs, state, reward, done, _ = env.step(state, action)

            #Aux-loss
            target = jnp.where(d_ghosts>=0, 
                               jnp.clip(1.0 - d_ghosts / DANGER_RADIUS,0.0,1.0), 0.0) * DANGER_MAX
            target = jax.lax.stop_gradient(target)
            aux = jnp.mean((danger - target) ** 2)

            # Ghost-Proximity Penalty
            pgx, pgy = snap(obs.player_position)                    
            agent_mask = jnp.zeros(walkable.shape, bool).at[pgx, pgy].set(True)
            dist = jax.lax.stop_gradient(plan(maze, agent_mask, walkable))
            ggx = (obs.ghost_positions[:, 0] + 5) // 4             
            ggy = (obs.ghost_positions[:, 1] + 3) // 4
            dangerous = state.ghosts.modes < 3
            nearest = jnp.min(jnp.where(dangerous, dist[ggx, ggy], LARGE_COST))
            prox_penalty = jnp.maximum(DANGER_RADIUS - nearest, 0.0)

            # Death Penalty 
            lives = state.lives
            died = (lives < prev_lives).astype(jnp.float32)
            
            reward = reward - DEATH_PENALTY * died - DANGER_WEIGHT * prox_penalty

            new_carry = (state, obs, prev_dir, lives, key)
            output = (logp, reward, done, aux)
            return new_carry, output

        final_carry, (logps, rewards, dones, auxs) = jax.lax.scan(step, init_carry, xs=None, length=episode_len)
        pg_loss, total_return = reinforce_loss(logps, rewards, dones)

        mask = ((jnp.cumsum(dones.astype(jnp.int32)) - dones.astype(jnp.int32)) == 0).astype(jnp.float32) 
        aux_loss = (auxs * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        loss = pg_loss + AUX_WEIGHT * aux_loss
        return loss, (total_return, aux_loss)

    def batch_loss(params, key):
        keys = jax.random.split(key, N_ENV)
        losses, (returns, auxs) = jax.vmap(episode_loss, in_axes=(None, 0))(params,keys)
        return losses.mean(), (returns.mean(), auxs.mean())

    @jax.jit
    def update(params, opt_state, key):
        (loss, (total_return, aux_loss)), grads = jax.value_and_grad(batch_loss, has_aux=True)(params,key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        return params, opt_state, loss, total_return, aux_loss, gnorm
        
    avg_return = None
    returns_log = []
    best_return = -1e9
    best_model = None
    fsave = True
    start = time.time()
    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss, total_return, aux_loss, gnorm = update(params, opt_state, sub)

        avg_return = float(total_return) if avg_return is None else 0.9 * avg_return + 0.1 * float(total_return)
        returns_log.append(float(total_return))

        if epoch % 10 == 0:
            elapsed = time.time() - start 
            per_epoch = elapsed / (epoch + 1)
            eta = per_epoch * (epochs - epoch - 1)
            print(f"epoch {epoch}  return {float(total_return):.0f}  avg {avg_return:.0f} aux {float(aux_loss):.2f}  gnorm {float(gnorm):.2f}  [ETA {eta/60:.1f} min]")

        if float(total_return) > best_return:
            best_return = float(total_return)
            best_model = params
            fsave = True

        if epoch % 100 == 0:
            save_params(params, f"outputs/ckpt_epoch{epoch}.msgpack")
            if fsave and best_model is not None:      # <-- guard
                save_params(best_model, "outputs/mspacman_best.msgpack")
                fsave = False

    params = best_model if best_model is not None else params
    save_params(params, "outputs/mspacman_params.msgpack")
    np.save("outputs/mspacman_returns.npy", np.array(returns_log))
    print(f"final return: {returns_log[-1]:.0f}  last_avgs: {np.mean(returns_log[-50:]):.0f}  best score: {float(best_return):.2f}")
    return params

# ----- Main -----
def main():
    from encoders.mspacman_encoder import make_mspacman_encoder
    env = jaxatari.make("mspacman")
    enc = make_mspacman_encoder(env)
    model = None
    params = train(env, enc, model, epochs=EPOCHS, )

if __name__ == "__main__":
    main()
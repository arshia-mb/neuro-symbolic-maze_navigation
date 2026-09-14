import argparse #TODO retire later

import jax
import jax.numpy as jnp
import numpy as np
import jaxatari

from src.bankheist.bh_encoder import bankheist_features
from .bh_train_policy import PolicyNet, load_policy, MAX_STEPS


def play_episode(env, params, apply_fn, key, render=False, max_steps=MAX_STEPS, verbose=True):
    obs, state = env.reset(key)
    frames = []
    total_reward = 0.0

    for t in range(max_steps):
        feat = bankheist_features(obs)
        logits = apply_fn(params, feat)
        action = int(jnp.argmax(logits))  # greedy 

        obs, state, reward, done, info = env.step(state, jnp.array(action))  # 5value return
        total_reward += float(reward)

        if render:
            frame = np.array(env.render(state))
            frames.append(frame)

        if verbose:
            print(f"  step {t:3d} | action={action:2d} | reward={float(reward):+.1f} | "
                  f"total={total_reward:+.1f} | lives={int(obs.lives)} | "
                  f"fuel={float(obs.fuel):.0f} | money={float(obs.score):.0f}")

        if bool(done):
            print(f"  episode ended at step {t} (done=True)")
            break

    return total_reward, frames


def main():
    parser = argparse.ArgumentParser() # recommended by claude
    parser.add_argument("--checkpoint", default="trained_bankheist_policy.pkl",
                         help="path to a checkpoint saved by train_bankheist_policy.py")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="suppress per-step logging")
    parser.add_argument("--render", action="store_true", help="save a GIF of the playthrough")
    parser.add_argument("--out", default="bankheist_playthrough.gif")
    args = parser.parse_args()

    env = jaxatari.make("bankheist")
    model = PolicyNet()
    params = load_policy(args.checkpoint)
    apply_fn = jax.jit(model.apply)

    all_returns = []
    all_frames = []

    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        print(f"\nepisode {ep} ")
        total_reward, frames = play_episode(env, params, apply_fn, key,
                                             render=args.render, verbose=not args.quiet)
        print(f"total_reward={total_reward:+.1f}")
        all_returns.append(total_reward)
        if args.render:
            all_frames.extend(frames)

    print(f"\nsum: {args.episodes} episodes")
    print(f"avg total_reward: {np.mean(all_returns):+.2f}")
    print(f"per episode:       {[f'{r:+.1f}' for r in all_returns]}")

    if args.render and all_frames:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in all_frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print(f"\nSaved playthrough GIF ({len(imgs)} frames) to {args.out}")


if __name__ == "__main__":
    main()
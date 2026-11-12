"""Sanity-check the custom Hopper with a random policy."""

import argparse

import gymnasium as gym
import numpy as np

from env import custom_hopper  # Registers the custom environments.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a random policy in a Custom Hopper environment."
    )
    parser.add_argument(
        "--env", default="CustomHopper-source-v0",
        help="Registered environment ID.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the environment in a human viewer.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    env = gym.make(
        args.env,
        render_mode="human" if args.render else None,
    )

    try:
        observation, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        print("Environment:", args.env)
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Dynamics parameters:", env.unwrapped.get_parameters())

        for episode in range(args.episodes):
            if episode > 0:
                observation, _ = env.reset(seed=args.seed + episode)

            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_return += float(reward)
                episode_length += 1

                if args.render:
                    env.render()

            print(
                f"Episode: {episode + 1} | "
                f"Length: {episode_length} | "
                f"Return: {episode_return:.2f} | "
                f"Terminated: {terminated} | "
                f"Truncated: {truncated}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
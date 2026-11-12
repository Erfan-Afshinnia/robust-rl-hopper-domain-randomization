"""Evaluate a custom REINFORCE or actor-critic Hopper policy."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from env import custom_hopper  # Registers the custom environments.
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved custom Hopper policy."
    )
    parser.add_argument(
        "--model", default="model.mdl", type=str,
        help="Path to a policy state_dict saved by train.py.",
    )
    parser.add_argument(
        "--env", default="CustomHopper-source-v0", type=str,
        help="Registered environment ID to evaluate.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str,
        help="Network device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the simulation in a human viewer.",
    )
    parser.add_argument(
        "--episodes", default=10, type=int,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed for evaluation.",
    )
    parser.add_argument(
        "--output-csv", default=None, type=str,
        help="Optional path for a per-episode results CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Policy file not found: {model_path.resolve()}. "
            "Train a custom policy first or specify --model."
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available. "
            "Use --device cpu or configure a CUDA-enabled PyTorch installation."
        )

    env = gym.make(
        args.env,
        render_mode="human" if args.render else None,
    )

    try:
        state, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        print("Dynamics parameters:", env.unwrapped.get_parameters())

        policy = Policy(
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
        )
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        policy.load_state_dict(state_dict, strict=True)

        agent = Agent(policy, device=args.device)

        rows = []
        for episode in range(args.episodes):
            if episode > 0:
                state, _ = env.reset(seed=args.seed + episode)

            episode_done = False
            episode_return = 0.0
            episode_length = 0

            while not episode_done:
                with torch.no_grad():
                    action, _ = agent.get_action(state, evaluation=True)

                action_np = action.detach().cpu().numpy()
                action_np = np.clip(
                    action_np,
                    env.action_space.low,
                    env.action_space.high,
                )

                state, reward, terminated, truncated, info = env.step(action_np)

                if args.render:
                    env.render()

                episode_return += float(reward)
                episode_length += 1
                episode_done = terminated or truncated

            rows.append({
                "environment": args.env,
                "episode": episode + 1,
                "seed": args.seed + episode,
                "return": episode_return,
                "length": episode_length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            print(
                f"Episode: {episode + 1} | "
                f"Return: {episode_return:.2f}"
            )

        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV saved: {output_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
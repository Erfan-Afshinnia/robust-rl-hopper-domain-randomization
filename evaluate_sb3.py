"""Evaluate saved PPO/SAC models and write per-episode CSV files."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

from env import custom_hopper  # Registers the custom environments.


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved PPO/SAC models without retraining."
    )
    parser.add_argument("--model", required=True, help="Path to saved .zip model")
    parser.add_argument("--algo", required=True, choices=["ppo", "sac"])
    parser.add_argument(
        "--envs", nargs="+", required=True,
        help="Environment IDs to evaluate on"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir", default="results/eval",
        help="Directory for per-episode CSV files"
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    model_class = PPO if args.algo == "ppo" else SAC
    model = model_class.load(str(model_path), device="cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_id in args.envs:
        rows = []
        env = gym.make(env_id)

        try:
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                total_reward = 0.0
                length = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    length += 1

                rows.append({
                    "model": model_path.name,
                    "algorithm": args.algo,
                    "env_id": env_id,
                    "episode": episode + 1,
                    "seed": args.seed + episode,
                    "reward": total_reward,
                    "length": length,
                    "terminated": terminated,
                    "truncated": truncated,
                })

        finally:
            env.close()

        safe_env_id = env_id.replace("/", "_").replace("\\", "_")
        output_path = output_dir / (
            f"{model_path.stem}__{safe_env_id}__eval{args.episodes}.csv"
        )

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        rewards = np.array([row["reward"] for row in rows], dtype=float)
        lengths = np.array([row["length"] for row in rows], dtype=float)
        print(f"\nModel: {model_path.name}")
        print(f"Environment: {env_id}")
        print(f"Episodes: {len(rows)}")
        print(f"Mean reward: {rewards.mean():.2f} +/- {rewards.std():.2f}")
        print(f"Mean length: {lengths.mean():.2f}")
        print(f"CSV saved: {output_path}")


if __name__ == "__main__":
    main()

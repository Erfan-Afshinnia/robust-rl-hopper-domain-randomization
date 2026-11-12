"""Train a custom policy on the Gymnasium Hopper environment."""

import argparse
import csv
import time
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch

from env import custom_hopper  # Registers the custom environments.
from agent import Agent, Policy, discount_rewards


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train REINFORCE or actor-critic on Custom Hopper."
    )
    parser.add_argument(
        "--n-episodes", default=100000, type=int,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--print-every", default=20000, type=int,
        help="Print training information every N episodes.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str,
        help="Network device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--baseline", default=None, type=float,
        help="Constant REINFORCE baseline; omit to disable.",
    )
    parser.add_argument(
        "--algo", default="reinforce",
        choices=("reinforce", "actor_critic"),
        help="Policy optimization algorithm.",
    )
    parser.add_argument(
        "--env", default="CustomHopper-source-v0", type=str,
        help="Registered environment ID.",
    )
    parser.add_argument(
        "--seed", default=42, type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-csv", type=str, default=None,
        help="Optional path for per-episode training metrics CSV.",
    )
    parser.add_argument(
        "--model-path", type=str, default="model.mdl",
        help="Path for saving the trained policy weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be greater than zero.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be greater than zero.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(args.env)

    try:
        # Seed the first reset and inspect the underlying custom environment.
        state, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        print("Dynamics parameters:", env.unwrapped.get_parameters())

        observation_space_dim = env.observation_space.shape[-1]
        action_space_dim = env.action_space.shape[-1]

        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(
            policy,
            device=args.device,
            baseline_value=args.baseline,
            algo=args.algo,
        )

        training_start = time.perf_counter()
        episode_rows = []
        for episode in range(args.n_episodes):
            episode_start = time.perf_counter()
            if episode > 0:
                state, _ = env.reset()

            episode_done = False
            train_reward = 0.0
            episode_rewards = []
            episode_length = 0

            while not episode_done:
                action, action_log_prob = agent.get_action(state)
                previous_state = state

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action.detach().cpu().numpy())

                # Treat genuine termination separately from a time-limit
                # truncation so actor-critic can bootstrap at truncation.
                agent.store_outcome(
                    previous_state,
                    next_state,
                    action_log_prob,
                    reward,
                    terminated,
                )

                state = next_state
                train_reward += reward
                episode_rewards.append(float(reward))
                episode_length += 1
                episode_done = terminated or truncated

            discounted_returns = discount_rewards(
                torch.as_tensor(episode_rewards, dtype=torch.float32),
                agent.gamma,
            ).cpu().numpy()

            loss = agent.update_policy()
            episode_rows.append({
                "episode": episode + 1,
                "return": train_reward,
                "mean_discounted_return": float(discounted_returns.mean()),
                "std_discounted_return": float(discounted_returns.std()),
                "length": episode_length,
                "loss": "" if loss is None else loss,
                "episode_seconds": time.perf_counter() - episode_start,
                "elapsed_seconds": time.perf_counter() - training_start,
            })

            if (episode + 1) % args.print_every == 0:
                print("Training episode:", episode + 1)
                print("Episode return:", train_reward)
                if loss is not None:
                    print("Loss:", loss)

        if args.log_csv:
            log_path = Path(args.log_csv)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(
                    log_file,
                    fieldnames=[
                        "episode", "return", "length", "loss", "mean_discounted_return", "std_discounted_return",
                        "episode_seconds", "elapsed_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerows(episode_rows)
            print(f"Saved episode metrics to {log_path.resolve()}")

        print(f"Training time: {time.perf_counter() - training_start:.2f} seconds")
        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.policy.state_dict(), model_path)
        print(f"Saved trained policy to {model_path.resolve()}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
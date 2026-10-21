"""Train and evaluate PPO or SAC on the custom Hopper environments."""

import argparse
from pathlib import Path

import gymnasium as gym
from env import custom_hopper  # Registers the custom environments.
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_vec_env(env_id: str, n_envs: int = 1, seed: int = 0):
    """Create vectorized Gymnasium environments with reproducible seeds."""

    def make_one(rank):
        def _init():
            env = gym.make(env_id)
            env.action_space.seed(seed + rank)
            return env
        return _init

    vec_env = DummyVecEnv([make_one(rank) for rank in range(n_envs)])
    vec_env.seed(seed)
    return vec_env


def train_ppo(
    train_env_id: str,
    total_timesteps: int = 200_000,
    device: str = "cpu",
    n_envs: int = 1,
    seed: int = 42,
):
    print(f"Creating PPO training environment: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs, seed=seed)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_hopper_tb/",
    )

    try:
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        print("PPO training finished.")
    except Exception:
        vec_env.close()
        raise

    return model


def train_sac(
    train_env_id: str,
    total_timesteps: int = 200_000,
    device: str = "cpu",
    n_envs: int = 1,
    seed: int = 42,
    learning_rate: float | None = None,
    batch_size: int | None = None,
):
    print(f"Creating SAC training environment: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs, seed=seed)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    sac_kwargs = {}
    if learning_rate is not None:
        sac_kwargs["learning_rate"] = learning_rate
    if batch_size is not None:
        sac_kwargs["batch_size"] = batch_size

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./sac_hopper_tb/",
        **sac_kwargs,
    )

    try:
        print(f"Starting SAC training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        print("SAC training finished.")
    except Exception:
        vec_env.close()
        raise

    return model


def evaluate_on_env(model, env_id: str, n_episodes: int = 50):
    """Evaluate a trained model on a standard Gymnasium environment."""
    print(f"\nEvaluating policy on environment: {env_id}")
    eval_env = Monitor(gym.make(env_id))

    try:
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_episodes,
            render=False,
            deterministic=True,
        )

        print(f"Evaluation over {n_episodes} episodes on {env_id}:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward
    finally:
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PPO or SAC on Custom Hopper."
    )
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=("ppo", "sac"),
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--train-env", type=str, default="CustomHopper-source-v0",
        help="Environment ID for training.",
    )
    parser.add_argument(
        "--test-env", type=str, default="CustomHopper-target-v0",
        help="Environment ID for transfer evaluation.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=200_000,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Training device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50,
        help="Episodes per built-in evaluation; 0 skips it.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Optional SAC learning rate; omitted uses the SB3 default.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Optional SAC batch size; omitted uses the SB3 default.",
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Model path without .zip; default keeps <algo>_hopper_<suffix>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be greater than zero.")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.eval_episodes < 0:
        raise ValueError("--eval-episodes must be zero or positive.")

    trainers = {"ppo": train_ppo, "sac": train_sac}
    trainer = trainers[args.algo]

    trainer_kwargs = {}
    if args.algo == "sac":
        trainer_kwargs["learning_rate"] = args.learning_rate
        trainer_kwargs["batch_size"] = args.batch_size

    model = trainer(
        train_env_id=args.train_env,
        total_timesteps=args.total_timesteps,
        device=args.device,
        n_envs=1,
        seed=args.seed,
        **trainer_kwargs,
    )

    suffix = (
        args.train_env.replace("CustomHopper-", "").replace("-v0", "")
        or "default"
    )
    save_name = args.save_path or f"{args.algo}_hopper_{suffix}"
    Path(save_name).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to: {save_name}.zip")
    model.save(save_name)

    try:
        if args.eval_episodes > 0:
            evaluate_on_env(model, args.train_env, n_episodes=args.eval_episodes)
            evaluate_on_env(model, args.test_env, n_episodes=args.eval_episodes)
    finally:
        if model.get_env() is not None:
            model.get_env().close()


if __name__ == "__main__":
    main()
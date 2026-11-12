"""Run a multi-condition SAC study for one training seed.

For every condition of the chosen study, the script trains SAC with
``train_sb3.py`` and evaluates the saved model on the standard source and
target environments with ``evaluate_sb3.py``. Existing models are reused, so
an interrupted run can be resumed by running the same command again.

Studies:
    udr       Uniform domain randomization of the non-torso masses
              (+/-10%, +/-30%, +/-50%) and the target-trained reference.
    ablation  2x2 ablation: {source, torso-mass DR} x {standard reward,
              height/posture reward}. Includes the source-trained baseline.
    hparam    SAC hyperparameter variants, tuned on the source domain only.

Examples:
    python run_study.py --study udr --seed 42
    python run_study.py --study ablation --seed 1 --steps 2000 --episodes 2 --out-dir results/smoke/ablation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

EVAL_ENVS = ("CustomHopper-source-v0", "CustomHopper-target-v0")

# Each condition: name -> (training environment, extra train_sb3.py arguments).
STUDIES = {
    "udr": {
        "udr10": ("CustomHopper-udr10-v0", []),
        "udr30": ("CustomHopper-udr-v0", []),
        "udr50": ("CustomHopper-udr50-v0", []),
        "tgt-std": ("CustomHopper-target-v0", []),
    },
    "ablation": {
        "src-std": ("CustomHopper-source-v0", []),
        "src-hp": ("CustomHopper-source-heightposture-v0", []),
        "tdr-std": ("CustomHopper-torsodr-v0", []),
        "tdr-hp": ("CustomHopper-torsodr-heightposture-v0", []),
    },
    "hparam": {
        "src-lr1e-4": ("CustomHopper-source-v0", ["--learning-rate", "0.0001"]),
        "src-ls10k": ("CustomHopper-source-v0", ["--learning-starts", "10000"]),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate every condition of a SAC study for one seed."
    )
    parser.add_argument("--study", required=True, choices=sorted(STUDIES))
    parser.add_argument("--seed", type=int, required=True, help="Training seed.")
    parser.add_argument(
        "--steps", type=int, default=200_000, help="SAC training timesteps."
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Evaluation episodes per environment."
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory; defaults to results/<study>_study or results/ablation.",
    )
    parser.add_argument("--device", default="cpu", help="Training device.")
    return parser.parse_args()


def default_out_dir(study):
    return Path("results/ablation" if study == "ablation" else f"results/{study}_study")


def run(cmd, log_path=None):
    """Run a command with the current interpreter and stop on failure."""
    if log_path is None:
        result = subprocess.run(cmd)
    else:
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        where = f" See {log_path}." if log_path else ""
        raise SystemExit(f"Command failed: {' '.join(cmd)}.{where}")


def main():
    args = parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        raise SystemExit("--steps and --episodes must be positive.")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.study)
    for sub in ("models", "eval", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for name, (train_env, extra) in STUDIES[args.study].items():
        tag = f"[{name} seed {args.seed}]"
        model_base = out_dir / "models" / f"sac_{name}_s{args.seed}"
        log_path = out_dir / "logs" / f"sac_{name}_s{args.seed}.log"

        if model_base.with_suffix(".zip").exists():
            print(f"{tag} model exists, skipping training.")
        else:
            print(f"{tag} training on {train_env} for {args.steps} steps {' '.join(extra)}".rstrip())
            start = time.perf_counter()
            run(
                [
                    sys.executable, "train_sb3.py", "--algo", "sac",
                    "--train-env", train_env, "--test-env", "CustomHopper-target-v0",
                    "--total-timesteps", str(args.steps), "--seed", str(args.seed),
                    "--device", args.device, "--eval-episodes", "0",
                    "--save-path", str(model_base), *extra,
                ],
                log_path,
            )
            print(f"{tag} training done in {(time.perf_counter() - start) / 60:.1f} min.")

        print(f"{tag} evaluating on source and target...")
        run(
            [
                sys.executable, "evaluate_sb3.py", "--model", str(model_base.with_suffix(".zip")),
                "--algo", "sac", "--envs", *EVAL_ENVS, "--episodes", str(args.episodes),
                "--seed", "1000", "--output-dir", str(out_dir / "eval"),
            ]
        )

    print(f"Seed {args.seed} finished. Aggregate with:")
    print(f"  python aggregate_seeds.py --eval-dir {out_dir / 'eval'} --output-dir {out_dir}")


if __name__ == "__main__":
    main()

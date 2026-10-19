# Robust Reinforcement Learning for Sim-to-Sim Transfer in MuJoCo Hopper

A reinforcement learning study of **robust control under dynamics mismatch** using a custom MuJoCo Hopper environment.

The project investigates how policies trained in one simulated domain transfer to another domain with different physical dynamics, and whether **domain randomization** improves transfer robustness. It combines from-scratch policy-gradient implementations in PyTorch with PPO and SAC baselines from Stable-Baselines3.

Built on the MLDL 2025 course starter code (lambdavi/rl_mldl_25, MIT license).

## Research Question

How does a change in simulated dynamics affect reinforcement learning performance, and can domain randomization improve zero-shot transfer between environments?

The custom environment defines source and target domains with different torso masses:

- **Source domain:** the torso mass is scaled to 70% of the target-domain value.
- **Target domain:** uses the original torso mass from the MuJoCo model.

This controlled dynamics mismatch provides the basis for the sim-to-sim transfer experiments.

## Methods

### From-Scratch Reinforcement Learning

Implemented in PyTorch:

- **REINFORCE:** Monte Carlo discounted returns, with an optional constant baseline.
- **Actor-Critic:** a learned state-value network, TD(0) targets, and advantage-based policy updates.
- **Gaussian policy:** tanh-squashed Gaussian actions with a learned per-action standard deviation and the corresponding log-probability correction, keeping actions within the environment bounds.

### Stable-Baselines3 Baselines

The repository also supports:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**

The Stable-Baselines3 training script supports configurable training and evaluation environments and evaluates policies on the selected training and test domains.

## Custom Domain-Randomization Environments

The environment uses Gymnasium and the official MuJoCo Python bindings.

| Environment | Configuration |
|---|---|
| `CustomHopper-v0` | Default target dynamics |
| `CustomHopper-source-v0` | Source dynamics |
| `CustomHopper-target-v0` | Target dynamics |
| `CustomHopper-udr-v0` | Uniform randomization of non-torso body masses |
| `CustomHopper-massdr-v0` | Mass-only ablation |
| `CustomHopper-frictiondr-v0` | Friction-only randomization |
| `CustomHopper-dampingdr-v0` | Damping-only randomization |
| `CustomHopper-extdr-v0` | Mass, damping, friction, and action noise |

The extended randomization condition combines mass randomization with multiplicative damping and friction changes and Gaussian action noise. Separate ablation environments allow individual parameter groups to be studied.

The observation contains 11 values: five position coordinates and six velocity coordinates. The action space contains three continuous controls bounded between -1 and 1. Registered environments have a maximum episode length of 500 steps.

## Installation

The working environment used for smoke tests was **Python 3.11.9 on Windows**. The pins in `requirements.txt` reflect the package versions tested in that environment.

From the repository root, open PowerShell and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To verify the installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

The project uses Gymnasium rather than legacy Gym and the official `mujoco` package rather than `mujoco-py`. The repository's Hopper XML has been converted to a format accepted by the tested MuJoCo version.

## Running the Training Scripts

Run commands from the repository root using the Python interpreter in `.venv`.

### REINFORCE

```powershell
.\.venv\Scripts\python.exe train.py --algo reinforce --env CustomHopper-source-v0
```

### Actor-Critic

```powershell
.\.venv\Scripts\python.exe train.py --algo actor_critic --env CustomHopper-source-v0
```

Both custom algorithms save their final policy state dictionary to `model.mdl`.

### PPO

```powershell
.\.venv\Scripts\python.exe train_sb3.py --algo ppo --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

### SAC

```powershell
.\.venv\Scripts\python.exe train_sb3.py --algo sac --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

The Stable-Baselines3 script saves the trained policy as a ZIP file and evaluates it on the training and test environments. TensorBoard logs are written to `ppo_hopper_tb/` or `sac_hopper_tb/`.

For shorter training runs, use `--total-timesteps` to reduce the number of steps. The default device is CPU; `--device cuda` can be used with a compatible CUDA-enabled PyTorch installation.

### Evaluate a Custom PyTorch Policy

```powershell
.\.venv\Scripts\python.exe test.py --model model.mdl --episodes 10 --env CustomHopper-source-v0
```

Replace the environment ID with `CustomHopper-target-v0` to evaluate in the target domain.

### Evaluate a Saved PPO/SAC Model

Use `evaluate_saved_models.py` to evaluate an existing Stable-Baselines3 model without retraining it. The script records per-episode results as CSV files.

For example, evaluate the saved source SAC model on both source and target environments for 50 episodes:

```powershell
.\.venv\Scripts\python.exe evaluate_saved_models.py --model sac_hopper_source_default_backup.zip --algo sac --envs CustomHopper-source-v0 CustomHopper-target-v0 --episodes 50 --seed 1000 --output-dir results/sb3_eval
```

Replace the model path and `--algo` value to evaluate a different saved PPO or SAC model. The model must already exist at the specified path.

### Test a Random Policy

Run a few episodes without rendering:

```powershell
.\.venv\Scripts\python.exe test_random_policy.py --episodes 5 --seed 42
```

Add `--render` to request a human-rendered simulation.

## Experimental Design

The project investigates:

- Source-to-source performance.
- Source-to-target zero-shot transfer.
- Target-to-target reference performance.
- Transfer robustness under domain randomization.
- Mass, friction, and damping ablations.
- SAC hyperparameter sensitivity across multiple random seeds.

### SAC Hyperparameter Search

The available results cover a **partial search** of four SAC configurations, with two seeds per configuration. Each run used 50,000 training steps and 10 evaluation episodes.

The best-performing tested configuration by mean return was:

| Parameter | Value |
|---|---:|
| Learning rate | `0.0001` |
| Batch size | `256` |
| Mean return across seeds | `339.03` |
| Standard deviation across seeds | `28.35` |

These results describe the configurations tested so far; they should not be interpreted as an exhaustive search or a guarantee of performance in other conditions. The result tables are in `hparam_search/`.

## Results and Visualizations

### PPO Baseline Comparison

![PPO baseline comparison](Bar%20Plots/ppo_baselines_barplot.png)

### SAC Robustness Comparison

![SAC robustness comparison](Bar%20Plots/sac_robustness_barplot.png)

### Simulation Demonstration

The repository includes an animation of a trained Hopper policy in a randomized environment.

![UDR Hopper simulation](GIF/hopper_udr_best.gif)

## Repository Structure

```text
robust-rl-hopper-domain-randomization/
|-- agent.py
|-- train.py
|-- train_sb3.py
|-- evaluate_saved_models.py
|-- test.py
|-- test_random_policy.py
|-- env/
|   |-- __init__.py
|   |-- custom_hopper.py
|   |-- mujoco_env.py
|   `-- assets/hopper.xml
|-- tests/
|   |-- test_custom_hopper.py
|   `-- test_agent.py
|-- hparam_search/
|   |-- hparam_results_partial.csv
|   `-- hparam_summary_partial.csv
|-- results/
|   |-- experiment_summary.md
|   |-- custom policy-gradient training/evaluation CSVs
|   `-- sb3_eval/ (50-episode evaluation CSVs)
|-- Bar Plots/
|-- GIF/
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```
## Development and Tests

Install the runtime and development dependencies in your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the automated environment tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The test suite checks the MuJoCo XML, all thirteen registered environment variants, source and target torso-mass properties, domain-randomization bounds and reproducibility, and the custom REINFORCE and Actor-Critic implementations.

## Experimental Findings

- In the corrected from-scratch experiments, REINFORCE with a constant baseline achieved the highest mean return over the final 100 training episodes across the two tested seeds. Actor-Critic had the shortest mean training time.
- In the 50-episode deterministic evaluations, all corrected custom policy-gradient policies terminated before reaching the 500-step limit. The baseline effect differed by seed, so these results do not establish a universally best algorithm.
- The partial SAC search favored learning rate `0.0001` and batch size `256` at 50,000 timesteps among the four tested configurations. The longer validation run did not outperform the recorded original default SAC baseline.
- The regenerated 50-episode UDR evaluation obtained a target mean reward of 901.44 +/- 25.59, compared with 686.63 +/- 2.60 for the default source-SAC backup evaluated on target. This is a difference in observed means, not proof that UDR is universally better; UDR performance varies across domains. Historical and regenerated evaluation batches remain separate.
- An additional UDR mass-range screen (seed 42, 50,000 training timesteps, 10 target episodes per configuration) tested +/-10%, +/-30%, and +/-50% ranges and obtained mean target rewards of 142.38, 382.45, and 354.57, respectively. The +/-30% setting had the highest observed mean in this preliminary single-seed screen, but all 30 episodes terminated before 500 steps. These shorter runs should not be compared directly with the main 200,000-step UDR evaluation.
- The extension policy reached the 500-step time limit in 99 of 100 regenerated source/target evaluation episodes. This supports sustained locomotion in those tested conditions, not universal robustness.

## Compatibility and Limitations

- The experiments concern **sim-to-sim transfer**, not real-world deployment or sim-to-real transfer.
- The migrated environment uses Gymnasium's reset and step interfaces and the official MuJoCo Python bindings.
- The converted XML's key physical properties were compared with the legacy model, and short simulation rollouts were close but not identical. Long-term trajectory equivalence has not been established.
- Short smoke tests verified that the environment variants, custom training algorithms, PPO, SAC, and policy evaluation execute with the migrated setup. These checks do not establish convergence or benchmark-quality performance.
- The hyperparameter-search results are partial and use two random seeds per configuration.

## Tech Stack

**Python, PyTorch, NumPy, Gymnasium, MuJoCo, Stable-Baselines3, PPO, SAC, REINFORCE, Actor-Critic, Domain Randomization**

## Author

**Erfan Afshinnia**

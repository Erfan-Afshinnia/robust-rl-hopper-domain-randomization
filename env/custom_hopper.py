"""Custom Hopper environment with domain randomization."""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils import EzPickle
from gymnasium.envs.registration import register, registry

from .mujoco_env import MujocoEnv


@dataclass
class MujocoState:
    """Small state container replacing the legacy MjSimState."""
    qpos: np.ndarray
    qvel: np.ndarray
    time: float


class CustomHopper(MujocoEnv, EzPickle):
    """Hopper with source, target, and randomized physics domains."""

    VALID_DOMAINS = {
        None, "source", "target", "udr", "massdr", "transferdr", "torsodr",
        "frictiondr", "dampingdr", "extdr",
    }

    def __init__(self, domain=None, render_mode=None, reward_mode="standard"):
        if domain not in self.VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain {domain!r}. "
                f"Choose one of {self.VALID_DOMAINS}."
            )

        if reward_mode not in ("standard", "posture", "posture_height"):
            raise ValueError(
                f"Unknown reward mode {reward_mode!r}; use 'standard', 'posture', or 'posture_height'."
            )

        self.domain = domain
        self.reward_mode = reward_mode
        self.action_noise_std = 0.01

        super().__init__(frame_skip=4, render_mode=render_mode)
        EzPickle.__init__(self, domain=domain, render_mode=render_mode, reward_mode=reward_mode)

        self.target_masses = self.model.body_mass[1:].copy()

        # Posture-reward reference and scales. The standard reward is unchanged.
        self.desired_height = float(self.init_qpos[1])
        self.height_penalty_weight = 0.5
        if self.reward_mode == "posture_height":
            self.height_penalty_weight = 2.0
        self.angle_penalty_weight = 0.5

        self.source_masses = self.target_masses.copy()
        self.source_masses[0] *= 0.7

        # Torso range defined relative to the source mass only (no target information).
        self.torso_scale_low = 0.5
        self.torso_scale_high = 1.5

        scale = 0.3
        self.udr_low = self.source_masses[1:] * (1.0 - scale)
        self.udr_high = self.source_masses[1:] * (1.0 + scale)

        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_friction = self.model.geom_friction.copy()

        self.damping_scale_low = 0.5
        self.damping_scale_high = 1.5
        self.friction_scale_low = 0.7
        self.friction_scale_high = 1.3

        self._set_observation_space(self._get_obs())

    def set_random_parameters(self):
        """Randomize the three non-torso body masses."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses using the original UDR ranges."""
        masses = self.source_masses.copy()
        masses[1:] = self.np_random.uniform(
            low=self.udr_low, high=self.udr_high
        )
        return masses

    def get_parameters(self):
        """Return the current masses of the four Hopper bodies."""
        return self.model.body_mass[1:].copy()

    def set_parameters(self, task):
        """Set the masses of the four Hopper bodies."""
        task = np.asarray(task, dtype=np.float64)
        if task.shape != self.target_masses.shape:
            raise ValueError(
                f"Expected mass shape {self.target_masses.shape}, "
                f"got {task.shape}"
            )
        self.model.body_mass[1:] = task

    def step(self, action):
        """Advance the simulation and return a Gymnasium transition."""
        action = np.asarray(action, dtype=np.float64)

        if self.domain == "extdr":
            action = action + self.np_random.normal(
                0.0, self.action_noise_std, size=action.shape
            )
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            )

        posbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, angle = self.data.qpos[0:3]

        reward = (posafter - posbefore) / self.dt
        reward += 1.0
        reward -= 1e-3 * np.square(action).sum()

        if self.reward_mode in ("posture", "posture_height"):
            # Normalize deviations so the penalty has a meaningful scale.
            height_error = (height - self.desired_height) / 0.25
            angle_error = angle / 0.2
            reward -= self.height_penalty_weight * height_error**2
            reward -= self.angle_penalty_weight * angle_error**2

        state = self.state_vector()
        terminated = not (
            np.isfinite(state).all()
            and (np.abs(state[2:]) < 100).all()
            and (height > 0.7)
            and (abs(angle) < 0.2)
        )

        observation = self._get_obs()
        return observation, float(reward), bool(terminated), False, {}

    def _get_obs(self):
        """Return five position coordinates and six velocity coordinates."""
        return np.concatenate([
            self.data.qpos[1:],
            self.data.qvel,
        ])

    def reset_model(self):
        """Reset the state and apply the selected physics domain."""
        rng = self.np_random

        qpos = self.init_qpos + rng.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + rng.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )

        self.model.dof_damping[:] = self.nominal_damping
        self.model.geom_friction[:] = self.nominal_friction

        if self.domain == "target" or self.domain is None:
            self.set_parameters(self.target_masses)

        elif self.domain == "source":
            self.set_parameters(self.source_masses)

        elif self.domain == "transferdr":
            # Oracle-range variant: torso sampled across the source-to-target interval.
            masses = self.target_masses.copy()
            masses[0] = rng.uniform(
                self.source_masses[0], self.target_masses[0]
            )
            self.set_parameters(masses)


        elif self.domain == "torsodr":
            # Leakage-free variant: torso range set relative to the source mass only.
            masses = self.source_masses.copy()
            masses[0] = rng.uniform(
                self.torso_scale_low * self.source_masses[0],
                self.torso_scale_high * self.source_masses[0],
            )
            self.set_parameters(masses)

        elif self.domain in ("udr", "massdr"):  # "massdr" is an alias of "udr"
            self.set_random_parameters()

        elif self.domain == "frictiondr":
            friction_scale = rng.uniform(
                self.friction_scale_low, self.friction_scale_high
            )
            self.model.geom_friction[:] = (
                self.nominal_friction * friction_scale
            )

        elif self.domain == "dampingdr":
            damping_scale = rng.uniform(
                self.damping_scale_low,
                self.damping_scale_high,
                size=self.nominal_damping.shape,
            )
            self.model.dof_damping[:] = (
                self.nominal_damping * damping_scale
            )

        elif self.domain == "extdr":
            self.set_random_parameters()

            damping_scale = rng.uniform(
                self.damping_scale_low,
                self.damping_scale_high,
                size=self.nominal_damping.shape,
            )
            self.model.dof_damping[:] = (
                self.nominal_damping * damping_scale
            )

            friction_scale = rng.uniform(
                self.friction_scale_low, self.friction_scale_high
            )
            self.model.geom_friction[:] = (
                self.nominal_friction * friction_scale
            )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure a wide tracking view of the complete Hopper."""
        if self.viewer is not None:
            torso_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = torso_id
            self.viewer.cam.distance = max(
                3.0, self.model.stat.extent * 2.0
            )
            self.viewer.cam.lookat[:] = self.data.xpos[torso_id]
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 90

    def set_mujoco_state(self, state):
        """Set state from the legacy 11-value environment-state vector."""
        state = np.asarray(state)
        if state.shape != (self.model.nq - 1 + self.model.nv,):
            raise ValueError(
                "Expected state with five position and six velocity values."
            )

        qpos = self.data.qpos.copy()
        qpos[0] = 0.0
        qpos[1:] = state[:self.model.nq - 1]
        qvel = state[self.model.nq - 1:]
        self.set_state(qpos, qvel)

    def get_mujoco_state(self):
        """Return copies of the current MuJoCo state."""
        return MujocoState(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            time=float(self.data.time),
        )

    def set_sim_state(self, state):
        """Restore a state returned by get_mujoco_state()."""
        if not isinstance(state, MujocoState):
            raise TypeError("state must be a MujocoState instance")
        self.set_state(state.qpos, state.qvel)
        self.data.time = state.time


_ENVIRONMENTS = {
    "CustomHopper-v0": None,
    "CustomHopper-source-v0": "source",
    "CustomHopper-target-v0": "target",
    "CustomHopper-udr-v0": "udr",
    "CustomHopper-massdr-v0": "massdr",   # alias of udr; identical behavior
    "CustomHopper-frictiondr-v0": "frictiondr",
    "CustomHopper-dampingdr-v0": "dampingdr",
    "CustomHopper-extdr-v0": "extdr",
}

for env_id, domain in _ENVIRONMENTS.items():
    if env_id not in registry:
        kwargs = {} if domain is None else {"domain": domain}
        register(
            id=env_id,
            entry_point="env.custom_hopper:CustomHopper",
            max_episode_steps=500,
            kwargs=kwargs,
        )

# Separate experimental environments with posture-aware reward shaping.
for env_id, domain in (
    ("CustomHopper-source-posture-v0", "source"),
    ("CustomHopper-target-posture-v0", "target"),
    ("CustomHopper-transferdr-posture-v0", "transferdr"),
):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="env.custom_hopper:CustomHopper",
            max_episode_steps=500,
            kwargs={"domain": domain, "reward_mode": "posture"},
        )

# Stronger height-penalty experiment; original environments stay unchanged.
for env_id, domain in (
    ("CustomHopper-transferdr-heightposture-v0", "transferdr"),
    ("CustomHopper-target-heightposture-v0", "target"),
):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="env.custom_hopper:CustomHopper",
            max_episode_steps=500,
            kwargs={"domain": domain, "reward_mode": "posture_height"},
        )

# Leakage-free 2x2 ablation of torso randomization and reward shaping.
for env_id, domain, reward_mode in (
    ("CustomHopper-torsodr-v0", "torsodr", "standard"),
    ("CustomHopper-source-heightposture-v0", "source", "posture_height"),
    ("CustomHopper-torsodr-heightposture-v0", "torsodr", "posture_height"),
):
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="env.custom_hopper:CustomHopper",
            max_episode_steps=500,
            kwargs={"domain": domain, "reward_mode": reward_mode},
        )
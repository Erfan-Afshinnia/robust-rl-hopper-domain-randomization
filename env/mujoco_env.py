"""Gymnasium-compatible base class for official MuJoCo environments."""

import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

DEFAULT_SIZE = 480


def convert_observation_to_space(observation):
    """Convert a NumPy observation into a Gymnasium space."""
    if isinstance(observation, dict):
        return spaces.Dict({
            key: convert_observation_to_space(value)
            for key, value in observation.items()
        })

    if isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -np.inf, dtype=observation.dtype)
        high = np.full(observation.shape, np.inf, dtype=observation.dtype)
        return spaces.Box(low=low, high=high, dtype=observation.dtype)

    raise NotImplementedError(
        f"Unsupported observation type: {type(observation).__name__}"
    )


class MujocoEnv(gym.Env):
    """Base class using the official MuJoCo Python bindings."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, frame_skip, render_mode=None):
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        if render_mode not in (None, "human", "rgb_array", "depth_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        self.viewer = None
        self._renderers = {}

        self.build_model()

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self._set_action_space()
        self.metadata = {
            **type(self).metadata,
            "render_fps": int(round(1.0 / self.dt)),
        }

    def build_model(self):
        """Load the MJCF model and create its simulation data."""
        model_path = os.path.join(
            os.path.dirname(__file__), "assets", "hopper.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.astype(np.float32, copy=True)
        self.action_space = spaces.Box(
            low=bounds[:, 0],
            high=bounds[:, 1],
            dtype=np.float32,
        )
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def reset(self, *, seed=None, options=None):
        """Reset the MuJoCo simulation and return observation and info."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        observation = self.reset_model()
        return observation, {}

    def reset_model(self):
        """Reset the robot state. Implement this in the subclass."""
        raise NotImplementedError

    def viewer_setup(self):
        """Optional subclass hook for viewer configuration."""
        pass

    def set_state(self, qpos, qvel):
        """Set joint positions and velocities."""
        qpos = np.asarray(qpos)
        qvel = np.asarray(qvel)

        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"Expected qpos shape {(self.model.nq,)}, got {qpos.shape}"
            )
        if qvel.shape != (self.model.nv,):
            raise ValueError(
                f"Expected qvel shape {(self.model.nv,)}, got {qvel.shape}"
            )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        """Apply an action for the requested number of simulation frames."""
        ctrl = np.asarray(ctrl)
        if ctrl.shape != (self.model.nu,):
            raise ValueError(
                f"Expected control shape {(self.model.nu,)}, got {ctrl.shape}"
            )

        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        """Render using the mode selected when the environment was created."""
        mode = self.render_mode
        if mode is None:
            return None

        if mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(
                    self.model, self.data
                )
                self.viewer_setup()
            self.viewer.sync()
            return None

        height = DEFAULT_SIZE
        width = DEFAULT_SIZE
        renderer = self._renderers.get((width, height))

        if renderer is None:
            renderer = mujoco.Renderer(
                self.model, height=height, width=width
            )
            self._renderers[(width, height)] = renderer

        if mode == "depth_array":
            renderer.enable_depth_rendering()

        try:
            renderer.update_scene(self.data, camera="track")
            image = renderer.render().copy()
        finally:
            if mode == "depth_array":
                renderer.disable_depth_rendering()

        return image

    def close(self):
        """Release viewer and renderer resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def get_body_com(self, body_name):
        """Return the world position of a named body."""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise ValueError(f"Unknown body name: {body_name!r}")
        return self.data.xpos[body_id].copy()

    def state_vector(self):
        """Return joint positions and velocities as one vector."""
        return np.concatenate([self.data.qpos, self.data.qvel])
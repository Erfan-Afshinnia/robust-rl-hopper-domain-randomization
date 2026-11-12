from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from env.custom_hopper import CustomHopper


ENV_IDS = [
    "CustomHopper-v0",
    "CustomHopper-source-v0",
    "CustomHopper-target-v0",
    "CustomHopper-udr-v0",
    "CustomHopper-massdr-v0",
    "CustomHopper-frictiondr-v0",
    "CustomHopper-dampingdr-v0",
    "CustomHopper-extdr-v0",
    "CustomHopper-source-posture-v0",
    "CustomHopper-target-posture-v0",
    "CustomHopper-transferdr-posture-v0",
    "CustomHopper-transferdr-heightposture-v0",
    "CustomHopper-target-heightposture-v0",
    "CustomHopper-torsodr-v0",
    "CustomHopper-source-heightposture-v0",
    "CustomHopper-torsodr-heightposture-v0",
    "CustomHopper-udr10-v0",
    "CustomHopper-udr50-v0",
]


def test_hopper_xml_compiles():
    xml_path = Path(__file__).parents[1] / "env" / "assets" / "hopper.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    assert model.nq == 6
    assert model.nv == 6
    assert model.nu == 3
    assert model.opt.timestep == pytest.approx(0.002)
    np.testing.assert_allclose(
        model.actuator_ctrlrange,
        np.tile([-1.0, 1.0], (3, 1)),
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_registered_environment_reset_and_step(env_id):
    env = gym.make(env_id)

    try:
        observation, info = env.reset(seed=42)

        assert observation.shape == (11,)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        result = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )

        assert len(result) == 5
        observation, reward, terminated, truncated, info = result

        assert observation.shape == (11,)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    finally:
        env.close()


def test_source_torso_mass_is_70_percent_of_target():
    target = CustomHopper(domain="target")
    source = CustomHopper(domain="source")

    try:
        target.reset(seed=42)
        source.reset(seed=42)

        target_mass = target.get_parameters()[0]
        source_mass = source.get_parameters()[0]

        assert source_mass == pytest.approx(0.7 * target_mass)
    finally:
        target.close()
        source.close()


def test_extdr_reset_is_reproducible():
    env = CustomHopper(domain="extdr")

    try:
        observation_1, _ = env.reset(seed=123)
        masses_1 = env.get_parameters()
        damping_1 = env.model.dof_damping.copy()
        friction_1 = env.model.geom_friction.copy()

        observation_2, _ = env.reset(seed=123)

        np.testing.assert_allclose(observation_1, observation_2)
        np.testing.assert_allclose(masses_1, env.get_parameters())
        np.testing.assert_allclose(damping_1, env.model.dof_damping)
        np.testing.assert_allclose(friction_1, env.model.geom_friction)
    finally:
        env.close()

def test_udr_preserves_source_torso_and_bounds_other_masses():
    env = CustomHopper(domain="udr")

    try:
        for seed in range(10):
            env.reset(seed=seed)
            masses = env.get_parameters()

            # Torso mass remains fixed at the source value.
            assert masses[0] == pytest.approx(env.source_masses[0])

            # Each of the other three masses stays within ±30%
            # of its corresponding source mass.
            assert np.all(masses[1:] >= env.udr_low)
            assert np.all(masses[1:] <= env.udr_high)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_id", "scale"),
    [
        ("CustomHopper-udr10-v0", 0.1),
        ("CustomHopper-udr-v0", 0.3),
        ("CustomHopper-udr50-v0", 0.5),
    ],
)
def test_udr_width_variants_use_their_scale(env_id, scale):
    env = gym.make(env_id)

    try:
        hopper = env.unwrapped
        np.testing.assert_allclose(
            hopper.udr_low, hopper.source_masses[1:] * (1.0 - scale)
        )
        np.testing.assert_allclose(
            hopper.udr_high, hopper.source_masses[1:] * (1.0 + scale)
        )

        for seed in range(10):
            env.reset(seed=seed)
            masses = hopper.get_parameters()
            assert masses[0] == pytest.approx(hopper.source_masses[0])
            assert np.all(masses[1:] >= hopper.udr_low)
            assert np.all(masses[1:] <= hopper.udr_high)
    finally:
        env.close()


@pytest.mark.parametrize("bad_scale", [0.0, 1.0, -0.1])
def test_invalid_udr_scale_is_rejected(bad_scale):
    with pytest.raises(ValueError):
        CustomHopper(domain="udr", udr_scale=bad_scale)

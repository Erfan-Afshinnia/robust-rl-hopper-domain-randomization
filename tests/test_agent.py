import numpy as np
import torch
import torch.nn.functional as F

from agent import Agent, Policy, discount_rewards


def test_discount_rewards_known_values():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    result = discount_rewards(rewards, gamma=0.5)
    expected = torch.tensor([2.75, 3.5, 3.0])
    assert torch.allclose(result, expected)


def test_initial_effective_policy_std_is_half():
    policy = Policy(state_space=11, action_space=3)
    std = F.softplus(policy.sigma)
    assert torch.allclose(std, torch.full((3,), 0.5), atol=1e-6)


def test_sampled_actions_are_bounded_and_log_probs_finite():
    torch.manual_seed(123)
    policy = Policy(state_space=11, action_space=3)
    agent = Agent(policy)
    state = np.zeros(11, dtype=np.float32)

    for _ in range(200):
        action, log_prob = agent.get_action(state)
        action_array = action.detach().cpu().numpy()

        assert action_array.shape == (3,)
        assert np.all(action_array >= -1.0)
        assert np.all(action_array <= 1.0)
        assert torch.isfinite(log_prob).item()


def test_deterministic_actions_are_bounded():
    policy = Policy(state_space=11, action_space=3)
    agent = Agent(policy)
    state = np.zeros(11, dtype=np.float32)

    action, log_prob = agent.get_action(state, evaluation=True)
    action_array = action.detach().cpu().numpy()

    assert log_prob is None
    assert np.all(action_array >= -1.0)
    assert np.all(action_array <= 1.0)


def _run_short_update(algo):
    torch.manual_seed(42)
    policy = Policy(state_space=11, action_space=3)
    agent = Agent(policy, algo=algo)

    states = [
        np.zeros(11, dtype=np.float32),
        np.full(11, 0.01, dtype=np.float32),
        np.full(11, 0.02, dtype=np.float32),
    ]

    for index, state in enumerate(states):
        action, log_prob = agent.get_action(state)
        next_state = states[index + 1] if index < 2 else state
        reward = float(index + 1)
        terminated = index == 2

        agent.store_outcome(
            state,
            next_state,
            log_prob,
            reward,
            terminated,
        )

    loss = agent.update_policy()
    assert loss is not None
    assert np.isfinite(loss)


def test_reinforce_update_has_finite_loss():
    _run_short_update("reinforce")


def test_reinforce_constant_baseline_update_has_finite_loss():
    torch.manual_seed(42)
    policy = Policy(state_space=11, action_space=3)
    agent = Agent(policy, baseline_value=1.5, algo="reinforce")

    states = [
        np.zeros(11, dtype=np.float32),
        np.full(11, 0.01, dtype=np.float32),
    ]

    for index, state in enumerate(states):
        action, log_prob = agent.get_action(state)
        next_state = states[index + 1] if index == 0 else state
        agent.store_outcome(
            state,
            next_state,
            log_prob,
            float(index + 1),
            index == 1,
        )

    loss = agent.update_policy()
    assert loss is not None
    assert np.isfinite(loss)


def test_actor_critic_update_has_finite_loss():
    _run_short_update("actor_critic")

def test_actor_critic_single_transition_has_finite_loss():
    torch.manual_seed(42)
    policy = Policy(state_space=11, action_space=3)
    agent = Agent(policy, algo="actor_critic")

    state = np.zeros(11, dtype=np.float32)
    action, log_prob = agent.get_action(state)

    agent.store_outcome(
        state=state,
        next_state=state,
        action_log_prob=log_prob,
        reward=1.0,
        done=True,
    )

    loss = agent.update_policy()

    assert loss is not None
    assert np.isfinite(loss)

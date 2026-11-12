"""Gaussian policy with a value head, plus REINFORCE and actor-critic updates."""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    """
    Compute discounted returns for a 1-D tensor of rewards r (shape [T]).
    Returns a tensor of the same shape containing discounted returns.
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor network: mean of a Gaussian over actions.
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learn a positive standard deviation through softplus.
        # Initialize the effective standard deviation to exactly 0.5.
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        raw_init_sigma = torch.log(torch.expm1(torch.tensor(init_sigma))).item()
        self.sigma = torch.nn.Parameter(
            torch.full((self.action_space,), raw_init_sigma)
        )

        # Critic network: state-value function used by actor-critic.
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        # Xavier initialization helps keep hidden-layer activations controlled.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Start with small action means to avoid immediate actuator saturation.
        torch.nn.init.uniform_(self.fc3_actor_mean.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.fc3_actor_mean.bias, -3e-3, 3e-3)


    def forward(self, x):
        # Actor forward pass.
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma) # learned per-action std
        # Normal supports broadcasting: action_mean shape [B, A], sigma shape [A]
        normal_dist = Normal(action_mean, sigma)
        # Critic forward pass.

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)  # shape: [batch] or []

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu', baseline_value=None, algo='reinforce'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.baseline_value = baseline_value  # used for REINFORCE with baseline
        self.algorithm = algo                 # 'reinforce' or 'actor_critic'

        # Buffers for a trajectory / episode
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        if len(self.action_log_probs) == 0:
            return  # nothing to update

        # Stack buffers and move to device
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear buffers
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if self.algorithm == 'reinforce':

            # Monte Carlo returns
            returns = discount_rewards(rewards, self.gamma)  # shape: [T]

            # Optional constant baseline.
            # Do not normalize after subtracting a constant: standardization
            # would cancel the baseline and make it ineffective.
            if self.baseline_value is not None:
                advantages = returns - self.baseline_value
            else:
                advantages = returns

            # REINFORCE policy gradient loss
            policy_loss = -(action_log_probs * advantages).sum()
            # 5. Backpropagation and optimizer step
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            return policy_loss.item()

        elif self.algorithm == 'actor_critic':

            # Get value estimates
            _, values = self.policy(states)          # shape: [T]
            _, next_values = self.policy(next_states)

            # TD(0) targets
            targets = rewards + self.gamma * (1.0 - done) * next_values.detach()

            # Advantages = TD error
            advantages = targets - values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            # Actor and critic losses
            actor_loss = -(action_log_probs * advantages.detach()).sum()
            critic_loss = F.mse_loss(values, targets.detach())

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
    


    def get_action(self, state, evaluation=False):
        """
        state: numpy array
        returns: (action, action_log_prob) where action_log_prob may be None in evaluation mode.
        """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)

        if evaluation:
            # Deterministic actions must respect the environment action bounds.
            return torch.tanh(normal_dist.mean), None

        # Sample in unconstrained space, then squash into (-1, 1).
        raw_action = normal_dist.sample()
        action = torch.tanh(raw_action)

        # Change-of-variables correction for the tanh transformation:
        # log pi(a) = log N(raw_action) - sum(log(1 - tanh(raw_action)^2)).
        log_prob_correction = torch.log(1.0 - action.pow(2) + 1e-6)
        action_log_prob = (
            normal_dist.log_prob(raw_action) - log_prob_correction
        ).sum()

        return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        # Store raw numpy states and convert to tensors during update
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


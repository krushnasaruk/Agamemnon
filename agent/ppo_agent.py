import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Any
from agent.base_agent import BaseRLAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 10, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Actor head (policy distribution)
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic head (state value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        features = self.feature_net(state)
        logits = self.actor(features)
        value = self.critic(features)
        dist = Categorical(logits=logits)
        return dist, value

class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization (PPO) Agent for Architecture Search.
    """
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 10,
        lr: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        mini_batch_size: int = 16,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.device = device

        self.ac_net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

        # Buffer for rollout trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.ac_net(state_tensor)
            if evaluate:
                action = torch.argmax(dist.logits, dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob=0.0, value=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def update(self) -> dict:
        if len(self.states) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        # Convert buffers to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = self.rewards
        dones = self.dones
        values = self.values + [0.0]

        # Compute GAE (Generalized Advantage Estimation)
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = advantages_t + torch.FloatTensor(values[:-1]).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        # Optimization epochs
        num_samples = len(self.states)
        indices = np.arange(num_samples)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, num_samples)
                mb_indices = indices[start_idx:end_idx]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                dist, current_values = self.ac_net(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ratio for PPO clip objective
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(current_values.squeeze(-1), mb_returns)

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # Clear trajectory buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

        return {
            "actor_loss": total_actor_loss / max(1, self.ppo_epochs),
            "critic_loss": total_critic_loss / max(1, self.ppo_epochs)
        }

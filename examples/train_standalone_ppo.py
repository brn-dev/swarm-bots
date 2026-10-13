"""Minimal PyTorch PPO training with SwarmBots' tensor environment API."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from gymnasium.vector import VectorEnv
import torch
from torch import nn
from torch.distributions import Normal

from swarmbots import make_env

Observations = Mapping[str, torch.Tensor]
Rollout = dict[str, torch.Tensor]


def policy_inputs(observations: Observations) -> tuple[torch.Tensor, torch.Tensor]:
    local_obs = observations["local_obs"]
    global_obs = observations["global_obs"].unsqueeze(1).expand(-1, local_obs.shape[1], -1)
    # Concatenation copies the inputs; the mask also needs a copy before the environment reuses it.
    return torch.cat((local_obs, global_obs), dim=-1), observations["agent_mask"].clone()


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, action_sizes: tuple[int, int]) -> None:
        super().__init__()
        self.action_sizes = action_sizes
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, sum(action_sizes))
        self.log_std = nn.Parameter(torch.zeros(sum(action_sizes)))
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: torch.Tensor, agent_mask: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.encoder(inputs)
        mask = agent_mask.unsqueeze(-1)
        pooled_features = features.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
        return Normal(self.actor(features), self.log_std.exp()), self.critic(pooled_features).squeeze(-1)

    def env_actions(self, raw_actions: torch.Tensor, agent_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        actions = raw_actions.tanh().masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        actuators, connectors = actions.split(self.action_sizes, dim=-1)
        return {"actuators": actuators, "connectors": connectors}

    @torch.no_grad()
    def act(self, observations: Observations, episode_starts: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs, agent_mask = policy_inputs(observations)
        distribution, _ = self(inputs, agent_mask)
        return self.env_actions(distribution.mean, agent_mask)


def joint_log_prob(distribution: Normal, raw_actions: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    # With stored pre-tanh samples, the tanh Jacobian cancels in PPO's old/new likelihood ratio.
    return distribution.log_prob(raw_actions).sum(dim=-1).masked_fill(~agent_mask, 0.0).sum(dim=-1)


@torch.no_grad()
def bootstrap_values(
    model: ActorCritic, observations: Observations, terminations: torch.Tensor,
    truncations: torch.Tensor, info: Mapping[str, Any],
) -> torch.Tensor:
    _, next_values = model(*policy_inputs(observations))
    if "final_obs" in info:
        # SAME_STEP returns reset observations; time limits need the terminal observations instead.
        truncated = info["_final_obs"] & truncations & ~terminations
        terminal_observations = {key: value[truncated] for key, value in info["final_obs"].items()}
        _, terminal_values = model(*policy_inputs(terminal_observations))
        next_values[truncated] = terminal_values
    return next_values.masked_fill(terminations, 0.0)


@torch.no_grad()
def collect_rollout(
    env: VectorEnv, model: ActorCritic, observations: dict[str, torch.Tensor], num_steps: int,
) -> tuple[Rollout, dict[str, torch.Tensor]]:
    storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    for _ in range(num_steps):
        inputs, agent_mask = policy_inputs(observations)
        distribution, values = model(inputs, agent_mask)
        raw_actions = distribution.sample()
        log_probs = joint_log_prob(distribution, raw_actions, agent_mask)
        observations, rewards, terminations, truncations, info = env.step(model.env_actions(raw_actions, agent_mask))
        transition = {
            "inputs": inputs, "agent_mask": agent_mask, "raw_actions": raw_actions,
            "log_probs": log_probs, "values": values, "rewards": rewards.clone(),
            "next_values": bootstrap_values(model, observations, terminations, truncations, info),
            "dones": terminations | truncations,
        }
        for key, value in transition.items():
            storage[key].append(value)
    return {key: torch.stack(values) for key, values in storage.items()}, observations


def compute_gae(rollout: Rollout, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.empty_like(rollout["rewards"])
    next_advantage = torch.zeros_like(advantages[0])
    for step in reversed(range(advantages.shape[0])):
        delta = rollout["rewards"][step] + gamma * rollout["next_values"][step] - rollout["values"][step]
        # Truncations bootstrap their value, but both episode boundaries stop the advantage trace.
        next_advantage = delta + gamma * gae_lambda * (~rollout["dones"][step]) * next_advantage
        advantages[step] = next_advantage
    return advantages, advantages + rollout["values"]


def update_policy(model: ActorCritic, optimizer: torch.optim.Optimizer, rollout: Rollout) -> None:
    advantages, returns = compute_gae(rollout)
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = returns.flatten()
    batch = {key: value.flatten(0, 1) for key, value in rollout.items()}
    for _ in range(4):
        indices = torch.randperm(returns.numel(), device=returns.device)
        for minibatch in indices.split(1024):
            distribution, values = model(batch["inputs"][minibatch], batch["agent_mask"][minibatch])
            log_probs = joint_log_prob(distribution, batch["raw_actions"][minibatch], batch["agent_mask"][minibatch])
            ratio = (log_probs - batch["log_probs"][minibatch]).exp()
            surrogate = ratio * advantages[minibatch]
            clipped_surrogate = ratio.clamp(0.8, 1.2) * advantages[minibatch]
            policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
            value_loss = 0.5 * (values - returns[minibatch]).square().mean()
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train(
    benchmark_id: str = "SwarmBots-WallEasy-v0", *, device: str = "cuda", num_envs: int = 256,
    total_timesteps: int = 1_000_000, rollout_steps: int = 128, seed: int = 42,
) -> ActorCritic:
    torch.manual_seed(seed)
    env = make_env(
        benchmark_id, num_envs=num_envs, device=device, seed=seed,
        scenario_kwargs={"compile_reward_kernel": False}, compile_tensor_operations=False,
    )
    try:
        input_size = sum(env.single_observation_space[key].shape[-1] for key in ("local_obs", "global_obs"))
        action_sizes = (env.single_action_space["actuators"].shape[-1], env.single_action_space["connectors"].shape[-1])
        model = ActorCritic(input_size, action_sizes).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        observations, _ = env.reset(seed=seed)
        timesteps = 0
        while timesteps < total_timesteps:
            remaining_steps = (total_timesteps - timesteps + num_envs - 1) // num_envs
            num_steps = min(rollout_steps, remaining_steps)
            rollout, observations = collect_rollout(env, model, observations, num_steps)
            update_policy(model, optimizer, rollout)
            timesteps += num_steps * num_envs
            print(f"steps={timesteps} mean_reward={float(rollout['rewards'].mean()):.3f}", flush=True)
        return model.eval()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_id", nargs="?", default="SwarmBots-WallEasy-v0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

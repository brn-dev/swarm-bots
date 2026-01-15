import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode
from swarmbots.learn.algos.ppo.wm.ppo_wm_sampler import PPOWMSampler


def _make_episode(
    *,
    num_steps: int,
    n_agents: int,
    n_local_obs_features: int,
    n_global_obs_features: int,
    n_actions: int,
    base: int,
) -> PPOEpisode:
    local_obs = (
        torch.arange(base, base + num_steps * n_agents * n_local_obs_features, dtype=torch.float32)
        .reshape(num_steps, n_agents, n_local_obs_features)
        .clone()
    )
    global_obs = (
        torch.arange(
            base + 100_000,
            base + 100_000 + num_steps * n_global_obs_features,
            dtype=torch.float32,
        )
        .reshape(num_steps, n_global_obs_features)
        .clone()
    )
    actions = (
        torch.arange(base + 200_000, base + 200_000 + num_steps * n_agents * n_actions, dtype=torch.float32)
        .reshape(num_steps, n_agents, n_actions)
        .clone()
    )

    final_local_obs = (
        torch.arange(
            base + 300_000,
            base + 300_000 + n_agents * n_local_obs_features,
            dtype=torch.float32,
        )
        .reshape(n_agents, n_local_obs_features)
        .clone()
    )

    return PPOEpisode(
        local_obs=local_obs,
        global_obs=global_obs,
        actions=actions,
        rewards=torch.zeros(num_steps, dtype=torch.float32),
        log_probs=torch.zeros((num_steps, n_agents), dtype=torch.float32),
        values=torch.zeros(num_steps, dtype=torch.float32),
        final_local_obs=final_local_obs,
        final_global_obs=torch.zeros(n_global_obs_features, dtype=torch.float32),
        final_value=torch.zeros(1, dtype=torch.float32),
        returns=torch.zeros(num_steps, dtype=torch.float32),
        advantages=torch.zeros(num_steps, dtype=torch.float32),
    )


def _expected_windows(
    *,
    local_obs: torch.Tensor,
    final_local_obs: torch.Tensor,
    actions: torch.Tensor,
    num_next_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps = local_obs.shape[0]

    next_obs = torch.cat([local_obs[1:], final_local_obs.unsqueeze(0)], dim=0)  # (T, N, F)

    expected_next_local_obs = torch.zeros(
        (num_steps, num_next_steps, *next_obs.shape[1:]),
        dtype=next_obs.dtype,
        device=next_obs.device,
    )
    expected_actions = torch.zeros(
        (num_steps, num_next_steps, *actions.shape[1:]),
        dtype=actions.dtype,
        device=actions.device,
    )
    expected_mask = torch.zeros((num_steps, num_next_steps), dtype=torch.bool, device=next_obs.device)

    for t in range(num_steps):
        for j in range(num_next_steps):
            src_idx = t + j
            if src_idx < num_steps:
                expected_next_local_obs[t, j] = next_obs[src_idx]
                expected_actions[t, j] = actions[src_idx]
                expected_mask[t, j] = True

    return expected_next_local_obs, expected_actions, expected_mask


def main() -> None:
    num_next_steps = 3

    ep1 = _make_episode(
        num_steps=4,
        n_agents=2,
        n_local_obs_features=2,
        n_global_obs_features=3,
        n_actions=2,
        base=0,
    )
    ep2 = _make_episode(
        num_steps=3,
        n_agents=2,
        n_local_obs_features=2,
        n_global_obs_features=3,
        n_actions=2,
        base=1_000_000,
    )

    sampler = PPOWMSampler([ep1, ep2], num_next_steps=num_next_steps)
    batch = sampler._fetch_samples(torch.arange(ep1.local_obs.shape[0] + ep2.local_obs.shape[0]))

    exp_next_1, exp_actions_1, exp_mask_1 = _expected_windows(
        local_obs=ep1.local_obs,
        final_local_obs=ep1.final_local_obs,
        actions=ep1.actions,
        num_next_steps=num_next_steps,
    )
    exp_next_2, exp_actions_2, exp_mask_2 = _expected_windows(
        local_obs=ep2.local_obs,
        final_local_obs=ep2.final_local_obs,
        actions=ep2.actions,
        num_next_steps=num_next_steps,
    )

    expected_next = torch.cat([exp_next_1, exp_next_2], dim=0)
    expected_actions = torch.cat([exp_actions_1, exp_actions_2], dim=0)
    expected_mask = torch.cat([exp_mask_1, exp_mask_2], dim=0)

    assert torch.equal(batch.next_local_obs, expected_next)
    assert torch.equal(batch.actions, expected_actions)
    assert torch.equal(batch.next_local_obs_mask, expected_mask)

    print("OK: PPOWMSampler multi-step next_obs/actions windows match expected values.")


if __name__ == "__main__":
    main()






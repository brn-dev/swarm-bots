import pytest
import torch
from unittest.mock import patch

from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.losses import mean_critic_mse_loss


@pytest.mark.parametrize("critic_count", [1, 2, 4])
def test_replicating_critic_heads_preserves_relative_auxiliary_loss_weight(critic_count: int) -> None:
    shared_prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    target = torch.zeros(2)
    auxiliary_loss = 0.1 * shared_prediction.square().mean()
    critic_loss = mean_critic_mse_loss([shared_prediction] * critic_count, target)
    total_loss = critic_loss + auxiliary_loss
    torch.testing.assert_close(total_loss, torch.tensor(5.5))
    total_loss.backward()
    torch.testing.assert_close(shared_prediction.grad, torch.tensor([1.1, 3.3]))


@pytest.mark.parametrize("kind", ["td3", "ddpg"])
def test_deterministic_learner_reports_mean_error_per_critic(kind: str) -> None:
    from tests.test_off_policy_commands import _make_algorithm
    from tests.test_sac_algorithm import _make_env

    env = _make_env()
    try:
        algorithm = _make_algorithm(env, kind)
        collect_off_policy_steps(env, algorithm.replay_buffer, n_steps=4, policy=algorithm.policy, random_actions=True)
        batch = algorithm.replay_buffer.sample(2)
        target = torch.ones_like(batch.rewards)
        with torch.no_grad():
            values = algorithm.policy.q_values(
                local_obs=batch.local_obs, global_obs=batch.global_obs, actions=batch.actions,
                hidden_local_vars=batch.hidden_local_vars, hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
            )
            expected_error = torch.cat([(value - target).square().flatten() for value in values]).mean().item()
        with patch.object(algorithm, "_bellman_target", return_value=target):
            metrics, _, _ = algorithm._train_step(batch, global_update_idx=0)
        assert metrics["critic_loss"] == pytest.approx(expected_error)
    finally:
        env.close()

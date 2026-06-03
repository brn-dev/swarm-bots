import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace


class _DummyActionDist:
    def __init__(self) -> None:
        self.distributions = [object()]
        self.has_gsde = False


class _DummyPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.Linear(3, 3, bias=False),
        )
        self.query_encoder = nn.Linear(3, 3, bias=False)
        self.context_encoder = nn.Linear(3, 3, bias=False)
        self.action_dist = _DummyActionDist()

    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, float]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "encoder": 0.0,
            "decoder": 0.0,
            "action_dist": 0.0,
            "critic": 0.0,
            "total": 0.0,
        }

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions, deterministic
        batch_size, n_agents = local_obs.shape[:2]
        return torch.zeros(batch_size, n_agents, 1, device=local_obs.device)

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unexpected weights: {weights}")

    def requires_previous_actions(self) -> bool:
        return False

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = hidden_local_vars, hidden_global_vars, agent_mask, previous_actions, deterministic
        batch_size, n_agents = local_obs.shape[:2]
        actions = torch.zeros(batch_size, n_agents, 1, device=local_obs.device)
        log_probs = torch.zeros(batch_size, n_agents, device=local_obs.device)
        values = torch.zeros(batch_size, device=local_obs.device)
        return actions, log_probs, values

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = local_obs, global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions
        return torch.zeros(1)

    def _evaluate_actions(
            self,
            batch,
            action_splitter: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
        _ = batch, action_splitter
        log_probs = torch.zeros(1)
        values = torch.zeros(1)
        local_latents = torch.zeros(1, 1, 1)
        return log_probs, values, {}, {}, local_latents

    def make_sampler(self, episodes, config):
        _ = episodes, config
        raise NotImplementedError


class _DummyEnv:
    def __init__(self) -> None:
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 3), dtype=np.float32),
            "global_obs": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
            "hidden_local_vars": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
            "hidden_global_vars": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
        })
        self.action_space = VectorHybridActionSpace({
            "move": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
        })


def _make_ppo(parameter_lr_multipliers: dict[str, float] | None = None, learning_rate: float = 1e-2) -> tuple[PPO, _DummyPolicy]:
    policy = _DummyPolicy()
    ppo = PPO(
        policy=policy,
        env=_DummyEnv(),
        learning_rate=learning_rate,
        rollout_mode=StepsRolloutMode(2),
        max_episode_length=2,
        n_epochs=1,
        parameter_lr_multipliers=parameter_lr_multipliers,
    )
    return ppo, policy


def _param_group_by_parameter_id(ppo: PPO) -> dict[int, dict[str, object]]:
    mapping: dict[int, dict[str, object]] = {}
    for group in ppo.optimizer.param_groups:
        for parameter in group["params"]:
            mapping[id(parameter)] = group
    return mapping


def test_ppo_uses_single_default_optimizer_group_when_no_multipliers_are_configured() -> None:
    ppo, _policy = _make_ppo()

    assert len(ppo.optimizer.param_groups) == 1
    assert ppo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert "lr_multiplier" not in ppo.optimizer.param_groups[0]


def test_parameter_lr_multipliers_scale_matching_parameters_and_keep_longest_prefix() -> None:
    ppo, policy = _make_ppo({
        "decoder": 0.5,
        "decoder.1": 0.2,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    })

    param_groups = _param_group_by_parameter_id(ppo)

    assert len(ppo.optimizer.param_groups) == 5
    assert param_groups[id(policy.shared.weight)]["lr"] == pytest.approx(1e-2)
    assert param_groups[id(policy.shared.weight)]["lr_multiplier"] == pytest.approx(1.0)
    assert param_groups[id(policy.decoder[0].weight)]["lr"] == pytest.approx(5e-3)
    assert param_groups[id(policy.decoder[0].weight)]["lr_multiplier"] == pytest.approx(0.5)
    assert param_groups[id(policy.decoder[1].weight)]["lr"] == pytest.approx(2e-3)
    assert param_groups[id(policy.decoder[1].weight)]["lr_multiplier"] == pytest.approx(0.2)
    assert param_groups[id(policy.query_encoder.weight)]["lr"] == pytest.approx(2.5e-3)
    assert param_groups[id(policy.query_encoder.weight)]["lr_multiplier"] == pytest.approx(0.25)
    assert param_groups[id(policy.context_encoder.weight)]["lr"] == pytest.approx(7.5e-3)
    assert param_groups[id(policy.context_encoder.weight)]["lr_multiplier"] == pytest.approx(0.75)

    ppo.set_learning_rate(2e-2)
    param_groups = _param_group_by_parameter_id(ppo)

    assert ppo.learning_rate == pytest.approx(2e-2)
    assert param_groups[id(policy.shared.weight)]["lr"] == pytest.approx(2e-2)
    assert param_groups[id(policy.decoder[0].weight)]["lr"] == pytest.approx(1e-2)
    assert param_groups[id(policy.decoder[1].weight)]["lr"] == pytest.approx(4e-3)
    assert param_groups[id(policy.query_encoder.weight)]["lr"] == pytest.approx(5e-3)
    assert param_groups[id(policy.context_encoder.weight)]["lr"] == pytest.approx(1.5e-2)


def test_optimizer_state_dict_round_trip_recovers_base_learning_rate() -> None:
    ppo, policy = _make_ppo({
        "decoder": 0.5,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    })
    optimizer_state = ppo._get_optimizer_state_dict()

    restored_ppo, restored_policy = _make_ppo({
        "decoder": 0.5,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    }, learning_rate=0.123)
    restored_ppo._apply_optimizer_state_dict(
        optimizer_state,
        missing_keys=[],
        unexpected_keys=[],
    )

    restored_groups = _param_group_by_parameter_id(restored_ppo)

    assert restored_ppo.learning_rate == pytest.approx(1e-2)
    assert restored_groups[id(restored_policy.shared.weight)]["lr"] == pytest.approx(1e-2)
    assert restored_groups[id(restored_policy.decoder[0].weight)]["lr"] == pytest.approx(5e-3)
    assert restored_groups[id(restored_policy.query_encoder.weight)]["lr"] == pytest.approx(2.5e-3)
    assert restored_groups[id(restored_policy.context_encoder.weight)]["lr"] == pytest.approx(7.5e-3)

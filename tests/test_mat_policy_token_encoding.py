import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_left_right_beta_action_dist import StickyLeftRightBetaConfig
from swarmbots.learn.algos.mat.mat_decoder import MATDecoderConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicy, MATPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyMATEnv:
    n_agents = 5
    local_obs_dim = 7
    global_obs_dim = 3
    hidden_local_vars_dim = 0
    hidden_global_vars_dim = 2
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


def _make_policy() -> MATPolicy:
    return MATPolicy(
        env=_DummyMATEnv(),
        config=MATPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                local_obs_encoder_hidden_dims=[16],
            ),
            decoder_config=MATDecoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                query_encoder_hidden_dims=[32],
                context_encoder_hidden_dims=[32],
                memory_dims=None,
            ),
            continuous_config=StickyLeftRightBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=False,
        ),
    )


def test_context_token_agent_embedding_slice_matches_full_sequence_encoding() -> None:
    policy = _make_policy()
    torch.manual_seed(0)

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATEnv.global_obs_dim)
    actions = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.action_space.total_agent_action_dim)

    with torch.no_grad():
        augmented_observations = policy.encoder(local_obs, global_obs)
        full_context_tokens = policy._encode_context_tokens(augmented_observations, actions)
        per_agent_context_tokens = torch.cat(
            [
                policy._encode_context_tokens(
                    augmented_observations[:, agent_idx:agent_idx + 1, :],
                    actions[:, agent_idx:agent_idx + 1, :],
                    agent_embeddings=policy.agent_embeddings_decoder[:, agent_idx:agent_idx + 1, :],
                )
                for agent_idx in range(_DummyMATEnv.n_agents)
            ],
            dim=1,
        )

    assert torch.allclose(full_context_tokens, per_agent_context_tokens, rtol=0.0, atol=1e-6)

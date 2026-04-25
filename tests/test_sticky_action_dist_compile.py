import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.action_dists.sticky_left_right_beta_action_dist import StickyLeftRightBetaConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


def _make_hybrid_dist(stickiness: float = 0.25) -> HybridActionDistribution:
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=float),
            "disc": spaces.MultiBinary((2, 1)),
        }
    )
    return HybridActionDistribution(
        latent_dim=8,
        action_space=action_space,
        continuous_config=StickyLeftRightBetaConfig(stickiness=stickiness),
        bernoulli_config=BernoulliConfig(initial_prob=0.5),
    )


def test_sticky_dist_always_requires_previous_actions() -> None:
    dist = _make_hybrid_dist()

    assert dist.requires_previous_actions() is True

    dist.set_sub_stickiness(0, 0.0)

    assert dist.requires_previous_actions() is True


def test_sticky_hybrid_log_prob_compiles_with_stickiness_annealed_to_zero() -> None:
    dist = _make_hybrid_dist()
    latent = torch.randn(4, 2, 8)
    previous_actions = torch.zeros(4, 2, dist.action_space.total_agent_action_dim)

    dist.update_latent_features(latent)
    actions = dist.get_actions(previous_actions=previous_actions)
    compiled_log_prob = torch.compile(dist.log_prob, backend="eager", fullgraph=False, dynamic=False)
    log_probs_before = compiled_log_prob(actions, previous_actions)

    dist.set_sub_stickiness(0, 0.0)
    dist.update_latent_features(latent)
    annealed_actions = dist.get_actions(previous_actions=previous_actions)
    log_probs_after = compiled_log_prob(annealed_actions, previous_actions)

    assert log_probs_before.shape == (4, 2)
    assert log_probs_after.shape == (4, 2)

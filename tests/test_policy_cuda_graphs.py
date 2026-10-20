from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.residual_mlp import ResidualMLPConfig


def _make_policy(variant: str, env: SimpleNamespace) -> BasePPOPolicy:
    encoder = MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    distribution = SignMagnitudeBetaConfig()
    if variant == "ppo":
        return PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(hidden_dims=[16], latent_pi_dim_per_agent=8),
            continuous_config=distribution,
        ))
    if variant.startswith("mappo"):
        return MAPPOPolicy(env, MAPPOPolicyConfig(
            actor_config=MAPPOActorConfig(
                latent_dim=8, latent_pi_dim=8,
                residual_mlp_config=ResidualMLPConfig(hidden_dim=16, num_blocks=1),
            ),
            actor_encoder_config=replace(encoder, use_agent_attention=False) if variant == "mappo_mat" else None,
            critic_encoder_config=encoder, continuous_config=distribution,
        ))
    return MATOrigPolicy(env, MATOrigPolicyConfig(
        encoder_config=encoder,
        decoder_config=MATOrigDecoderConfig(d_model=8, nhead=2, num_layers=1, latent_pi_dim=8),
        continuous_config=distribution,
    ))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("variant", ["ppo", "mappo", "mappo_mat", "mat_orig"])
def test_cuda_graph_inference_preserves_outputs_across_policy_calls(variant: str) -> None:
    env = SimpleNamespace(
        n_agents=2, local_obs_dim=3, global_obs_dim=1, hidden_local_vars_dim=1, hidden_global_vars_dim=1,
        action_space=HybridActionSpace({"control": spaces.Box(-1.0, 1.0, (2, 2))}),
    )
    eager = _make_policy(variant, env).cuda().eval()
    compiled = type(eager)(env, replace(eager.config, compile_modules=True, compile_mode="reduce-overhead"))
    compiled = compiled.cuda().eval()
    compiled.load_state_dict({
        key: eager.state_dict()[key.replace("_orig_mod.", "")]
        for key in compiled.state_dict()
    })
    observations = {
        "local_obs": torch.randn(2, 2, 3, device="cuda"),
        "global_obs": torch.randn(2, 1, device="cuda"),
        "hidden_local_vars": torch.randn(2, 2, 1, device="cuda"),
        "hidden_global_vars": torch.randn(2, 1, device="cuda"),
        "agent_mask": torch.tensor([[True, True], [True, False]], device="cuda"),
    }
    retained = []
    with torch.no_grad():
        for offset in (0.0, 0.25, 0.5):
            current = {**observations, "local_obs": observations["local_obs"] + offset}
            expected = eager(**current, deterministic=True)
            actual = compiled(**current, deterministic=True)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            retained.append((actual, tuple(tensor.clone() for tensor in actual)))
            for method in ("predict_values", "act"):
                options = {"deterministic": True} if method == "act" else {}
                expected_result = getattr(eager, method)(**current, **options)
                result = getattr(compiled, method)(**current, **options)
                torch.testing.assert_close(result, expected_result, rtol=1e-4, atol=1e-5)
                retained.append((result, result.clone()))
        for result, snapshot in retained:
            torch.testing.assert_close(result, snapshot, rtol=0, atol=0)

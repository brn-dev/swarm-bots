import unittest
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any
from unittest.mock import patch

import torch

import swarmbots.learn.algos.sac.sac_nop as sac_nop
from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.sac_nop import SACNOPConfig, SACNOPModule
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig


_REAL_TORCH_COMPILE = torch.compile


def _compile_with_aot_eager(
        function: Callable[..., Any],
        **kwargs: Any,
) -> Callable[..., Any]:
    return _REAL_TORCH_COMPILE(
        function,
        backend="aot_eager",
        fullgraph=kwargs["fullgraph"],
        dynamic=kwargs["dynamic"],
    )


def _nop_config(*, compile_modules: bool = False, nop_loss_coef: float = 1.0) -> SACNOPConfig:
    return SACNOPConfig(
        enabled=True,
        num_next_steps=3,
        nop_latent_dim=8,
        nop_loss_coef=nop_loss_coef,
        compile_modules=compile_modules,
        transition_model_d_model=8,
        transition_model_nhead=2,
        transition_model_num_layers=1,
        transition_model_dim_feedforward=16,
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=[0],
            local_angle_target_indices=[1],
            local_rot6d_target_indices=[3],
            local_binary_target_indices=[9],
            global_scalar_target_indices=[0],
            global_rot6d_target_indices=[1],
            predict_delta=False,
        ),
    )


def _make_module(
        *,
        config: SACNOPConfig | None = None,
        skip_first_transition: bool = False,
) -> SACNOPModule:
    return SACNOPModule(
        n_agents=3,
        source_latent_dim=6,
        action_dim=2,
        config=_nop_config() if config is None else config,
        name="test",
        skip_first_transition=skip_first_transition,
    )


def _set_structured_targets(local_obs: torch.Tensor, global_obs: torch.Tensor) -> None:
    local_angles = torch.randn(local_obs.shape[:-1])
    local_obs[..., 1] = local_angles.sin()
    local_obs[..., 2] = local_angles.cos()
    local_obs[..., 9] = torch.randint(
        0,
        2,
        local_obs[..., 9].shape,
        device=local_obs.device,
        dtype=torch.long,
    ).to(dtype=local_obs.dtype)
    global_obs[..., 1:7] = torch.randn_like(global_obs[..., 1:7])


def _make_flat_batch(batch_size: int = 4) -> OffPolicyReplayBatch:
    n_agents = 3
    local_obs = torch.randn(batch_size, n_agents, 10)
    next_local_obs = torch.randn_like(local_obs)
    global_obs = torch.randn(batch_size, 7)
    next_global_obs = torch.randn_like(global_obs)
    _set_structured_targets(local_obs, global_obs)
    _set_structured_targets(next_local_obs, next_global_obs)
    mask_repeats = (batch_size + 3) // 4
    agent_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ]).repeat(mask_repeats, 1)[:batch_size]
    next_agent_mask = torch.tensor([
        [True, True, True],
        [True, False, False],
        [True, False, False],
        [True, True, False],
    ]).repeat(mask_repeats, 1)[:batch_size]
    return OffPolicyReplayBatch(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=torch.empty(batch_size, n_agents, 0),
        hidden_global_vars=torch.empty(batch_size, 0),
        agent_mask=agent_mask,
        actions=torch.randn(batch_size, n_agents, 2),
        rewards=torch.zeros(batch_size),
        terminations=torch.zeros(batch_size, dtype=torch.bool),
        truncations=torch.zeros(batch_size, dtype=torch.bool),
        previous_actions=None,
        next_local_obs=next_local_obs,
        next_global_obs=next_global_obs,
        next_hidden_local_vars=torch.empty(batch_size, n_agents, 0),
        next_hidden_global_vars=torch.empty(batch_size, 0),
        next_agent_mask=next_agent_mask,
    )


def _make_segment_batch(batch_size: int = 2, sequence_length: int = 3) -> OffPolicyReplayEpisodeSegmentBatch:
    flat_batch = _make_flat_batch(batch_size * sequence_length)

    def reshape(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])

    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=reshape(flat_batch.local_obs),
        global_obs=reshape(flat_batch.global_obs),
        hidden_local_vars=reshape(flat_batch.hidden_local_vars),
        hidden_global_vars=reshape(flat_batch.hidden_global_vars),
        agent_mask=reshape(flat_batch.agent_mask),
        actions=reshape(flat_batch.actions),
        rewards=reshape(flat_batch.rewards),
        terminations=reshape(flat_batch.terminations),
        truncations=reshape(flat_batch.truncations),
        previous_actions=None,
        next_local_obs=reshape(flat_batch.next_local_obs),
        next_global_obs=reshape(flat_batch.next_global_obs),
        next_hidden_local_vars=reshape(flat_batch.next_hidden_local_vars),
        next_hidden_global_vars=reshape(flat_batch.next_hidden_global_vars),
        next_agent_mask=reshape(flat_batch.next_agent_mask),
        episode_start_mask=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        train_mask=torch.tensor([[True, True, False], [True, False, False]]),
        initial_temporal_state=None,
        burn_in_steps=0,
    )


def _move_batch_to_cuda(batch: OffPolicyReplayBatch) -> OffPolicyReplayBatch:
    return replace(
        batch,
        **{
            field.name: value.cuda() if isinstance(value, torch.Tensor) else value
            for field in fields(batch)
            if (value := getattr(batch, field.name)) is not None
        },
    )


class SACNOPTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA compilation")
    def test_cuda_inductor_full_graph_matches_eager_loss_and_backward(self) -> None:
        torch.manual_seed(5)
        eager_module = _make_module().cuda().eval()
        compiled_module = _make_module(
            config=_nop_config(compile_modules=True),
        ).cuda().eval()
        compiled_module.load_state_dict(eager_module.state_dict())
        batch = _move_batch_to_cuda(_make_flat_batch(batch_size=2))
        eager_latents = torch.randn(2, 3, 6, device="cuda", requires_grad=True)
        compiled_latents = eager_latents.detach().clone().requires_grad_(True)

        eager_loss, _eager_metrics = eager_module.compute_loss(
            source_latents=eager_latents,
            batch=batch,
        )
        compiled_loss, _compiled_metrics = compiled_module.compute_loss(
            source_latents=compiled_latents,
            batch=batch,
        )
        eager_loss.backward()
        compiled_loss.backward()

        torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(compiled_latents.grad, eager_latents.grad, rtol=1e-4, atol=1e-5)
        for eager_parameter, compiled_parameter in zip(
                eager_module.parameters(),
                compiled_module.parameters(),
                strict=True,
        ):
            torch.testing.assert_close(
                compiled_parameter.grad,
                eager_parameter.grad,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_constructor_rejects_invalid_dimensions_weights_and_missing_targets(self) -> None:
        valid_config = _nop_config()
        invalid_cases = (
            ({"source_latent_dim": 0, "action_dim": 2, "config": valid_config}, "source_latent_dim"),
            ({"source_latent_dim": 6, "action_dim": 0, "config": valid_config}, "action_dim"),
            ({
                "source_latent_dim": 6,
                "action_dim": 2,
                "config": replace(valid_config, nop_loss_coef=-0.1),
            }, "nop_loss_coef"),
            ({
                "source_latent_dim": 6,
                "action_dim": 2,
                "config": replace(valid_config, nop_latent_dim=0),
            }, "nop_latent_dim"),
            ({
                "source_latent_dim": 6,
                "action_dim": 2,
                "config": replace(valid_config, next_obs_pred_config=NextObsPredConfig()),
            }, "prediction target"),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SACNOPModule(n_agents=3, name="invalid", **kwargs)

    def test_compilation_uses_one_end_to_end_full_graph(self) -> None:
        with patch.object(
                sac_nop.torch,
                "compile",
                side_effect=lambda function, **_kwargs: function,
        ) as compile_mock:
            module = _make_module(config=_nop_config(compile_modules=True))

        compile_mock.assert_called_once()
        compile_call = compile_mock.call_args
        self.assertEqual(compile_call.args[0].__name__, "_compute_next_obs_pred_loss_impl")
        self.assertEqual(
            compile_call.kwargs,
            {"mode": "default", "fullgraph": True, "dynamic": False},
        )
        self.assertIsInstance(module.transition_model, torch.nn.Module)
        self.assertFalse(hasattr(module.transition_model, "_orig_mod"))

    def test_compile_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "compile_mode"):
            _make_module(config=replace(_nop_config(), compile_modules=True, compile_mode=""))
        with patch.object(sac_nop.torch, "compile", None):
            with self.assertRaisesRegex(RuntimeError, "torch.compile support"):
                _make_module(config=_nop_config(compile_modules=True))

    def test_compiled_all_target_loss_matches_eager_outputs_metrics_and_gradients(self) -> None:
        torch.manual_seed(10)
        eager_module = _make_module()
        torch.manual_seed(20)
        with patch.object(
                sac_nop.torch,
                "compile",
                side_effect=_compile_with_aot_eager,
        ):
            compiled_module = _make_module(config=_nop_config(compile_modules=True))
        compiled_module.load_state_dict(eager_module.state_dict())
        batch = _make_flat_batch()
        eager_latents = torch.randn(4, 3, 6, requires_grad=True)
        compiled_latents = eager_latents.detach().clone().requires_grad_(True)

        eager_loss, eager_metrics = eager_module.compute_loss(
            source_latents=eager_latents,
            batch=batch,
        )
        compiled_loss, compiled_metrics = compiled_module.compute_loss(
            source_latents=compiled_latents,
            batch=batch,
        )
        eager_loss.backward()
        compiled_loss.backward()

        torch.testing.assert_close(compiled_loss, eager_loss)
        self.assertEqual(compiled_metrics.keys(), eager_metrics.keys())
        for key in eager_metrics:
            self.assertAlmostEqual(compiled_metrics[key], eager_metrics[key], places=6)
        torch.testing.assert_close(compiled_latents.grad, eager_latents.grad)
        for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
                eager_module.named_parameters(),
                compiled_module.named_parameters(),
                strict=True,
        ):
            self.assertEqual(compiled_name, eager_name)
            self.assertIsNotNone(eager_parameter.grad, eager_name)
            self.assertIsNotNone(compiled_parameter.grad, compiled_name)
            torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)
        torch.testing.assert_close(compiled_module.binary_target_ema, eager_module.binary_target_ema)

    def test_sequence_masks_exclude_padded_times_and_inactive_next_agents(self) -> None:
        torch.manual_seed(30)
        module = _make_module()
        module.eval()
        batch = _make_segment_batch()
        source_latents = torch.randn(2, 3, 6)
        baseline_loss, _metrics = module.compute_loss(
            source_latents=source_latents,
            batch=batch,
        )

        invalid_time_mask = ~batch.train_mask.unsqueeze(-1).unsqueeze(-1)
        inactive_agent_mask = ~batch.next_agent_mask.unsqueeze(-1)
        modified_next_local_obs = batch.next_local_obs.masked_fill(
            invalid_time_mask | inactive_agent_mask,
            10_000.0,
        )
        modified_next_global_obs = batch.next_global_obs.masked_fill(
            ~batch.train_mask.unsqueeze(-1),
            -10_000.0,
        )
        modified_batch = replace(
            batch,
            next_local_obs=modified_next_local_obs,
            next_global_obs=modified_next_global_obs,
        )
        modified_loss, _metrics = module.compute_loss(
            source_latents=source_latents,
            batch=modified_batch,
        )

        torch.testing.assert_close(modified_loss, baseline_loss)

    def test_loss_coefficient_remains_mutable_outside_compiled_graph(self) -> None:
        with patch.object(
                sac_nop.torch,
                "compile",
                side_effect=_compile_with_aot_eager,
        ):
            module = _make_module(config=_nop_config(compile_modules=True, nop_loss_coef=1.0))
        module.eval()
        batch = _make_flat_batch()
        source_latents = torch.randn(4, 3, 6)
        original_loss, original_metrics = module.compute_loss(
            source_latents=source_latents,
            batch=batch,
        )

        module.nop_loss_coef = 0.25
        scaled_loss, scaled_metrics = module.compute_loss(
            source_latents=source_latents,
            batch=batch,
        )

        torch.testing.assert_close(scaled_loss, original_loss * 0.25)
        self.assertAlmostEqual(
            scaled_metrics["test_nop_loss"],
            original_metrics["test_nop_loss"],
        )
        self.assertAlmostEqual(
            scaled_metrics["test_nop_loss_scaled"],
            original_metrics["test_nop_loss_scaled"] * 0.25,
        )

    def test_skip_first_transition_supports_single_and_multi_step_batches(self) -> None:
        module = _make_module(skip_first_transition=True)
        flat_batch = _make_flat_batch()
        flat_loss, _metrics = module.compute_loss(
            source_latents=torch.randn(4, 3, 6),
            batch=flat_batch,
        )
        segment_batch = _make_segment_batch()
        segment_loss, _metrics = module.compute_loss(
            source_latents=torch.randn(2, 3, 6),
            batch=segment_batch,
        )

        self.assertTrue(torch.isfinite(flat_loss))
        self.assertTrue(torch.isfinite(segment_loss))


if __name__ == "__main__":
    unittest.main()

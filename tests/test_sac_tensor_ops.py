import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch

import swarmbots.learn.algos.sac.sac_tensor_ops as sac_tensor_ops
from swarmbots.learn.algos.sac.sac_tensor_ops import (
    build_optimizer_step,
    build_sac_tensor_operations,
)


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


class SACTensorOperationsTests(unittest.TestCase):
    def test_eager_operations_match_explicit_sac_equations(self) -> None:
        operations = build_sac_tensor_operations(
            compile_operations=False,
            compile_mode="default",
        )
        log_probs = torch.tensor([
            [-3.0, -6.0, 20.0],
            [-2.0, -4.0, -8.0],
        ])
        agent_mask = torch.tensor([
            [True, True, False],
            [True, False, False],
        ])

        log_prob_mean = operations.mean_agent_log_probs(log_probs, agent_mask)
        torch.testing.assert_close(log_prob_mean, torch.tensor([-4.5, -2.0]))
        torch.testing.assert_close(
            operations.mean_agent_log_probs(log_probs, None),
            log_probs.mean(dim=-1),
        )

        rewards = torch.tensor([1.0, 2.0])
        terminal_mask = torch.tensor([True, False])
        target_q1 = torch.tensor([torch.nan, 5.0])
        target_q2 = torch.tensor([torch.nan, 4.0])
        ent_coef = torch.tensor(0.25)
        target = operations.bellman_target(
            rewards,
            terminal_mask,
            target_q1,
            target_q2,
            torch.tensor([torch.nan, -2.0]),
            ent_coef,
            0.9,
        )
        torch.testing.assert_close(target, torch.tensor([1.0, 6.05]))

        current_q1 = torch.tensor([0.0, 1.0])
        current_q2 = torch.tensor([2.0, 3.0])
        expected_critic_loss = 0.5 * (
            torch.nn.functional.mse_loss(current_q1, target)
            + torch.nn.functional.mse_loss(current_q2, target)
        )
        torch.testing.assert_close(
            operations.critic_loss(current_q1, current_q2, target),
            expected_critic_loss,
        )
        expected_actor_loss = (
            ent_coef * log_prob_mean - torch.minimum(current_q1, current_q2)
        ).mean()
        torch.testing.assert_close(
            operations.actor_loss(current_q1, current_q2, log_prob_mean, ent_coef),
            expected_actor_loss,
        )

    def test_terminal_masking_covers_every_bootstrap_field_and_attention_mask(self) -> None:
        operations = build_sac_tensor_operations(
            compile_operations=False,
            compile_mode="default",
        )
        terminal_mask = torch.tensor([True, False])
        local_obs = torch.full((2, 3, 4), torch.nan)
        global_obs = torch.full((2, 2), torch.nan)
        hidden_local_vars = torch.full((2, 3, 1), torch.nan)
        hidden_global_vars = torch.full((2, 1), torch.nan)
        agent_mask = torch.tensor([
            [False, False, False],
            [True, False, True],
        ])
        for tensor in (local_obs, global_obs, hidden_local_vars, hidden_global_vars):
            tensor[1] = 7.0

        masked = operations.mask_terminal_observations(
            terminal_mask,
            local_obs,
            global_obs,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask,
        )

        for tensor in masked[:-1]:
            self.assertTrue(torch.equal(tensor[0], torch.zeros_like(tensor[0])))
            self.assertTrue(torch.equal(tensor[1], torch.full_like(tensor[1], 7.0)))
        assert masked[-1] is not None
        self.assertEqual(masked[-1].tolist(), [[True, True, True], [True, False, True]])

    def test_compiled_operations_match_eager_outputs_and_gradients(self) -> None:
        with patch.object(
                sac_tensor_ops.torch,
                "compile",
                side_effect=_compile_with_aot_eager,
        ) as compile_mock:
            compiled = build_sac_tensor_operations(
                compile_operations=True,
                compile_mode="reduce-overhead",
            )
        eager = build_sac_tensor_operations(
            compile_operations=False,
            compile_mode="default",
        )

        self.assertEqual(compile_mock.call_count, 6)
        for call in compile_mock.call_args_list:
            self.assertEqual(
                call.kwargs,
                {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False},
            )

        base_inputs = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([False, True]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([2.0, 5.0]),
            torch.tensor([-0.5, -1.0]),
            torch.tensor(0.2),
            0.95,
        )
        torch.testing.assert_close(
            compiled.bellman_target(*base_inputs),
            eager.bellman_target(*base_inputs),
        )

        for operation_name in ("critic_loss", "actor_loss"):
            eager_inputs = [
                torch.tensor([1.0, 2.0], requires_grad=True),
                torch.tensor([3.0, 4.0], requires_grad=True),
                torch.tensor([0.5, -1.0], requires_grad=True),
            ]
            compiled_inputs = [value.detach().clone().requires_grad_(True) for value in eager_inputs]
            if operation_name == "critic_loss":
                eager_loss = eager.critic_loss(*eager_inputs)
                compiled_loss = compiled.critic_loss(*compiled_inputs)
            else:
                eager_loss = eager.actor_loss(*eager_inputs, torch.tensor(0.3))
                compiled_loss = compiled.actor_loss(*compiled_inputs, torch.tensor(0.3))
            eager_loss.backward()
            compiled_loss.backward()
            torch.testing.assert_close(compiled_loss, eager_loss)
            for eager_input, compiled_input in zip(eager_inputs, compiled_inputs, strict=True):
                torch.testing.assert_close(compiled_input.grad, eager_input.grad)

        eager_log_ent_coef = torch.tensor(-1.0, requires_grad=True)
        compiled_log_ent_coef = eager_log_ent_coef.detach().clone().requires_grad_(True)
        log_prob_mean = torch.tensor([-2.0, -3.0], requires_grad=True)
        target_entropy = torch.tensor([-1.0, -1.0])
        eager_entropy_loss = eager.entropy_coefficient_loss(
            eager_log_ent_coef,
            log_prob_mean,
            target_entropy,
        )
        compiled_entropy_loss = compiled.entropy_coefficient_loss(
            compiled_log_ent_coef,
            log_prob_mean.detach().clone().requires_grad_(True),
            target_entropy,
        )
        eager_entropy_loss.backward()
        compiled_entropy_loss.backward()
        torch.testing.assert_close(compiled_entropy_loss, eager_entropy_loss)
        torch.testing.assert_close(compiled_log_ent_coef.grad, eager_log_ent_coef.grad)
        self.assertIsNone(log_prob_mean.grad)

    def test_compile_configuration_is_validated(self) -> None:
        with patch.object(sac_tensor_ops.torch, "compile", None):
            with self.assertRaisesRegex(RuntimeError, "torch.compile support"):
                build_sac_tensor_operations(
                    compile_operations=True,
                    compile_mode="default",
                )
        with self.assertRaisesRegex(ValueError, "sac_compile_mode"):
            build_sac_tensor_operations(
                compile_operations=True,
                compile_mode="",
            )

    def test_optimizer_step_stays_eager_when_compilation_is_disabled(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        with patch.object(sac_tensor_ops.torch, "compile") as compile_mock:
            step = build_optimizer_step(
                optimizer,
                compile_step=False,
                compile_mode="default",
            )
        self.assertEqual(step, optimizer.step)
        compile_mock.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA optimizer compilation")
    def test_compiled_adam_step_matches_eager_across_stateful_updates(self) -> None:
        eager_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
        compiled_parameter = torch.nn.Parameter(eager_parameter.detach().clone())
        eager_optimizer = torch.optim.Adam([eager_parameter], lr=3e-3)
        compiled_optimizer = torch.optim.Adam([compiled_parameter], lr=3e-3)
        compiled_step = build_optimizer_step(
            compiled_optimizer,
            compile_step=True,
            compile_mode="default",
        )

        for target in (0.5, -1.5, 2.0):
            eager_optimizer.zero_grad()
            compiled_optimizer.zero_grad()
            eager_parameter.sub(target).square().sum().backward()
            compiled_parameter.sub(target).square().sum().backward()
            eager_optimizer.step()
            compiled_step()

            torch.testing.assert_close(compiled_parameter, eager_parameter)
            eager_state = eager_optimizer.state[eager_parameter]
            compiled_state = compiled_optimizer.state[compiled_parameter]
            self.assertEqual(eager_state.keys(), compiled_state.keys())
            for key in eager_state:
                torch.testing.assert_close(compiled_state[key].cpu(), eager_state[key].cpu())


if __name__ == "__main__":
    unittest.main()

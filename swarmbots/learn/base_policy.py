import abc
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import torch
from torch import nn

from swarmbots.learn.temporal_state import clone_temporal_state


PolicyInferenceArgs = ParamSpec("PolicyInferenceArgs")
PolicyInferenceResult = TypeVar("PolicyInferenceResult")


class PolicyCompileConfig(Protocol):
    @property
    def compile_modules(self) -> bool: ...

    @property
    def compile_mode(self) -> str: ...


class CompiledPolicy(Protocol):
    @property
    def config(self) -> PolicyCompileConfig: ...


PolicyInferenceOwner = TypeVar("PolicyInferenceOwner", bound=CompiledPolicy)


def policy_inference_step(
    method: Callable[Concatenate[PolicyInferenceOwner, torch.Tensor, PolicyInferenceArgs], PolicyInferenceResult],
) -> Callable[Concatenate[PolicyInferenceOwner, torch.Tensor, PolicyInferenceArgs], PolicyInferenceResult]:
    @wraps(method)
    def inference_step(
        policy: PolicyInferenceOwner,
        local_obs: torch.Tensor,
        *args: PolicyInferenceArgs.args,
        **kwargs: PolicyInferenceArgs.kwargs,
    ) -> PolicyInferenceResult:
        config = policy.config
        uses_cuda_graphs = (
            local_obs.is_cuda
            and not torch.is_grad_enabled()
            and config.compile_modules
            and config.compile_mode in ("reduce-overhead", "max-autotune")
        )
        if not uses_cuda_graphs:
            return method(policy, local_obs, *args, **kwargs)
        # Composed compiled modules share a graph iteration; returned tensors
        # must survive the next policy call, including bootstrap predictions.
        torch.compiler.cudagraph_mark_step_begin()
        return clone_temporal_state(method(policy, local_obs, *args, **kwargs))

    return inference_step


class BasePolicy(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def gsde_enabled(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_hyper_parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_grad_norms(self) -> dict[str, float]:
        raise NotImplementedError()

    @abc.abstractmethod
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def update_loss_weights(self, **weights: float) -> None:
        """
        Updates weights for extra losses. If weights contains an unknown key, a ValueError is thrown
        :param weights:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def requires_previous_actions(self) -> bool:
        raise NotImplementedError()

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> Any:
        _ = batch_size
        _ = n_agents
        _ = device
        _ = dtype
        return None

    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: Any = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        _ = episode_start_mask
        action_kwargs = dict(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )
        if scenario_ids is not None:
            action_kwargs["scenario_ids"] = scenario_ids
        actions = self.act(**action_kwargs)
        return actions, temporal_state

    @staticmethod
    def _grad_norm_from_parameters(parameters: Iterable[nn.Parameter]) -> float:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return 0.0
        return float(nn.utils.get_total_norm(gradients, norm_type=2.0).item())

    @classmethod
    def _module_grad_norm(cls, module: nn.Module | None) -> float:
        if module is None:
            return 0.0
        return cls._grad_norm_from_parameters(module.parameters())

    @classmethod
    def _parameter_grad_norm(cls, parameter: nn.Parameter | None) -> float:
        if parameter is None:
            return 0.0
        return cls._grad_norm_from_parameters((parameter,))

    def num_parameters(self, learnable_only: bool = True) -> int:
        if learnable_only:
            return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.parameters())

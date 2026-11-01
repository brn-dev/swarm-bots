from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from tests.test_base_algorithm_lifecycle import _Algorithm
from tests.test_live_hyperparameter_updates import algorithm as algorithm, single_thread as single_thread


def test_mutation_commands_cannot_change_any_learner(algorithm: BaseAlgorithm) -> None:
    before = deepcopy(algorithm.get_hyper_parameters())
    parameters = {name: value.clone() for name, value in algorithm.policy.state_dict().items()}
    commands = (
        "set_lr:0.002", "set_std:0.2", "scale_std:2", "set_n_epochs:2", "set_num_epochs:2",
        "set_target_kl:0.02", "set_clip_range:0.2", "set_clip_range_vf:0.2", "set_mc_ent_coef:0.1",
        "set_vf_coef:0.1", "set_batch_size:4", "batch_size:4", "set_normalize_advantage:false",
        "normalize_advantage:false", "set_gsde_prob:0.2", "set_gsde_probability:0.2", "gsde_prob:0.2",
        "set_gsde_interval:2", "gsde_interval:2", "disable_gsde_reset", "gsde_reset_off",
        "set_max_grad_norm:0.5", "max_grad_norm:0.5", "set_gamma:0.5", "set_gae_lambda:0.5",
        'set_extra_loss_weights:{"scalar":2}', 'set_loss_weights:{"scalar":2}', 'loss_weights:{"scalar":2}',
        "set_wm_loss_coef:0.5", "set_world_model_loss_coef:0.5", "wm_loss_coef:0.5", "set_nop_loss_coef:0.5",
        "set_wm_num_next_steps:2", "set_world_model_num_next_steps:2", "wm_num_next_steps:2",
        "disable_auto_lr", "auto_lr_off", "disable_automatic_lr", "disable_auto_learning_rate",
        "enable_auto_lr", "auto_lr_on", "enable_automatic_lr", "enable_auto_learning_rate",
        "set_tau:0.5", "set_ent_coef:0.5", "ent_coef:0.5", 'set_rw:{"progress_reward_weight":2}',
        'set_reward_weights:{"progress_reward_weight":2}', "save:unexpected.pt", "record:3",
        "pause:1h", "unpause", "resume", "stop", "quit", "exit",
    )
    with patch.object(algorithm, "save") as save:
        for command in commands:
            assert not algorithm.execute_command(command)
            if ":" in command:
                assert not algorithm.execute_command(command.split(":", 1)[0])
        save.assert_not_called()

    assert algorithm.get_hyper_parameters() == before
    assert (algorithm.n_total_iterations, algorithm.n_total_updates, algorithm.n_total_timesteps) == (0, 0, 0)
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize("command,key", (
    ("show_hps", "learning_rate"), ("show_run_path", "run_path"), ("show_id", "run_id"),
    ("pid", "pid"), ("show_pid", "pid"), ("show_reward_weights", "reward_weights"),
    ("show_rw", "reward_weights"), ("eta", "estimated_finish_at"), ("show_eta", "estimated_finish_at"),
    ("record_status", "recording"), ("record_progress", "recording"),
    ("recording_status", "recording"), ("recording_progress", "recording"),
))
def test_inspection_commands_report_state_without_mutating_it(command: str, key: str) -> None:
    algorithm = _Algorithm()
    algorithm.env = SimpleNamespace(unwrapped=SimpleNamespace(
        get_reward_weights=lambda: {"progress_reward_weight": 1.0},
        get_video_recording_status=lambda: {"active": False},
    ))
    before = dict(vars(algorithm))

    with patch("swarmbots.learn.algos.base_algorithm.logger.info") as log:
        assert algorithm.execute_command(f"  {command.upper()}  ")

    assert key in log.call_args.args[0]
    assert vars(algorithm) == before


def test_inspection_queue_continues_after_an_unknown_or_failed_command() -> None:
    algorithm = _Algorithm()
    algorithm._command_queue.put("unknown; show_rw; ; pid; show_hps")

    with patch("swarmbots.learn.algos.base_algorithm.logger.info") as log:
        algorithm._poll_and_execute_commands()

    assert algorithm._command_queue.empty()
    assert log.call_count == 2
    assert "pid" in log.call_args_list[0].args[0]
    assert "learning_rate" in log.call_args_list[1].args[0]

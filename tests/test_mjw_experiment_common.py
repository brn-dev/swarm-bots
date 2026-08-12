import inspect
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
import pytest
import torch
from torch import nn

import experiments.mjw_find_opening_tmasac.scripts.common as find_opening_common
import experiments.mjw_experiment_common as experiment_common
from experiments.mjw_experiment_common import (
    MATInitGains,
    MATNormalizationConfig,
    PolicyVariant,
    _make_base_policy,
    _make_mat_parameter_lr_multipliers,
    _make_staggered_first_episode_lengths,
    _run_training_with_notification_and_close,
    _with_default_scenario_kwargs,
    set_actuator_gsde_init_joint_stds,
    wrap_vec_env,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
    GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
)
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaActionDist
from swarmbots.learn.action_dists.ternary_sign_magnitude_beta_action_dist import (
    TernarySignMagnitudeBetaActionDist,
)
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSPolicy
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
    RecurrentTMASACPolicy,
)
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_actor_heads import (
    TMASACActorHeadKind,
    TMASACQCXActorHead,
)
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import (
    SwarmBotsLearnEnvWrapper,
)
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.env_wrappers.torch_progress_guidance_ep_stats_wrapper import (
    TorchProgressGuidanceEpisodeStatsWrapper,
)
from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import (
    TorchRecordEpisodeStatisticsWrapper,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.feed_forward import GLUStackConfig, MLPConfig, StackedGLU, SwiGLUConfig
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


class _DummyEnv:
    n_agents = 3
    local_obs_dim = 4
    global_obs_dim = 2
    hidden_local_vars_dim = 0
    hidden_global_vars_dim = 0
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


class _DummyContinuousEnv:
    n_agents = 3
    local_obs_dim = 4
    global_obs_dim = 2
    hidden_local_vars_dim = 0
    hidden_global_vars_dim = 0
    action_space = HybridActionSpace(
        {
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "connectors": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 1), dtype=float),
        }
    )


class _FakeExperimentVectorEnv:
    single_observation_space = {
        "local_obs": SimpleNamespace(shape=(3, 4)),
        "global_obs": SimpleNamespace(shape=(2,)),
        "hidden_local_vars": SimpleNamespace(shape=(3, 0)),
        "hidden_global_vars": SimpleNamespace(shape=(0,)),
    }

    def get_settings(self) -> dict[str, object]:
        return {"test_setting": True}


class _FakeExperimentEnv:
    n_agents = 3
    local_obs_dim = 4
    global_obs_dim = 2
    actuators_dim = 2
    connectors_dim = 1
    continuous_connector_actions = True
    action_space = SimpleNamespace(total_agent_action_dim=3)

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeExperimentPolicy:
    local_latent_dim = 8
    action_dist = SimpleNamespace(distributions=[object()])

    def num_parameters(self) -> int:
        return 1


class _FakeExperimentAlgorithm:
    def __init__(self, capture: dict[str, object]) -> None:
        self.capture = capture

    def learn(self, **kwargs: object) -> None:
        self.capture["learn_kwargs"] = kwargs


def _make_obs_indices() -> ObsIndices:
    return ObsIndices(
        local_scalar_indices=[0, 1],
        local_angle_indices=[],
        local_rot6d_indices=[],
        local_binary_indices=[],
        local_quaternion_indices=[],
        global_scalar_indices=[],
        global_rot6d_indices=[],
        global_quaternion_indices=[],
        hidden_local_vars_scalar_indices=[],
        hidden_local_vars_quaternion_indices=[],
        hidden_global_vars_scalar_indices=[],
        hidden_global_vars_quaternion_indices=[],
    )


def _make_wrapper_test_vector_env() -> SyncVectorEnv:
    return SyncVectorEnv(
        [
            lambda: TestingSwarmBotsEnv(
                n_agents=3,
                n_local_obs=4,
                n_global_obs=2,
                actuators_dim=2,
                connectors_dim=1,
                continuous_connector_actions=True,
            )
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def _patch_default_experiment_boundaries(
        monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], _FakeExperimentEnv]:
    capture: dict[str, object] = {}
    vector_env = _FakeExperimentVectorEnv()
    env = _FakeExperimentEnv()
    policy = _FakeExperimentPolicy()
    algorithm = _FakeExperimentAlgorithm(capture)
    recording_hook = object()

    monkeypatch.setattr(
        experiment_common,
        "logger",
        SimpleNamespace(
            remove=Mock(),
            add=Mock(),
            info=Mock(),
            warning=Mock(),
            error=Mock(),
        ),
    )
    monkeypatch.setattr(experiment_common, "_get_requested_cuda_idx", lambda: None)
    monkeypatch.setattr(experiment_common, "_configure_cuda_device", lambda **_kwargs: None)
    monkeypatch.setattr(experiment_common, "configure_float32_matmul_precision", lambda: None)
    monkeypatch.setattr(experiment_common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "swarmbots.learn.torch_logging.enable_torch_compile_logging",
        lambda: None,
    )

    def make_vector_env(**kwargs: object) -> _FakeExperimentVectorEnv:
        capture["vector_env_kwargs"] = kwargs
        return vector_env

    def wrap_vector_env(**kwargs: object) -> _FakeExperimentEnv:
        capture["wrap_vec_env_kwargs"] = kwargs
        return env

    def make_base_policy(**kwargs: object) -> _FakeExperimentPolicy:
        capture["base_policy_kwargs"] = kwargs
        return policy

    def wrap_policy(*, policy: object, world_model_config: object) -> object:
        capture["world_model_config"] = world_model_config
        return policy

    def make_ppo(**kwargs: object) -> _FakeExperimentAlgorithm:
        capture["ppo_kwargs"] = kwargs
        return algorithm

    def make_sac(**kwargs: object) -> _FakeExperimentAlgorithm:
        capture["sac_kwargs"] = kwargs
        return algorithm

    def make_recurrent_sac(**kwargs: object) -> _FakeExperimentAlgorithm:
        capture["recurrent_sac_kwargs"] = kwargs
        return algorithm

    def install_recordings(**kwargs: object) -> object:
        capture["recording_kwargs"] = kwargs
        return recording_hook

    def run_notification(*, run: Callable[[], None], **kwargs: object) -> None:
        capture["notification_kwargs"] = kwargs
        run()

    monkeypatch.setattr(experiment_common, "make_vector_env", make_vector_env)
    monkeypatch.setattr(experiment_common, "build_obs_indices", lambda **_kwargs: _make_obs_indices())
    monkeypatch.setattr(experiment_common, "wrap_vec_env", wrap_vector_env)
    monkeypatch.setattr(experiment_common, "_make_base_policy", make_base_policy)
    monkeypatch.setattr(experiment_common, "NextObsPredWrapper", wrap_policy)
    monkeypatch.setattr(experiment_common, "set_actuator_gsde_init_joint_stds", lambda **_kwargs: None)
    monkeypatch.setattr(experiment_common, "PPO", make_ppo)
    monkeypatch.setattr(experiment_common, "SAC", make_sac)
    monkeypatch.setattr(experiment_common, "RecurrentSAC", make_recurrent_sac)
    monkeypatch.setattr(experiment_common, "install_scheduled_recordings", install_recordings)
    monkeypatch.setattr(experiment_common, "run_with_discord_notification", run_notification)
    capture["algorithm"] = algorithm
    capture["recording_hook"] = recording_hook
    return capture, env


def _make_test_base_policy(
        *,
        policy_variant: PolicyVariant,
        mat_normalization: MATNormalizationConfig = MATNormalizationConfig(),
        **overrides: object,
) -> object:
    kwargs: dict[str, object] = {
        "env": _DummyEnv(),
        "policy_variant": policy_variant,
        "enc_d_model": 8,
        "enc_nhead": 1,
        "dec_d_model": 8,
        "dec_nhead": 1,
        "use_popart": False,
        "popart_beta": 0.1,
        "popart_init_sigma": 1.0,
        "compile_policy_modules": False,
        "policy_compile_mode": "default",
        "continuous_action_dist": "sign_magnitude_beta",
        "initial_stickiness": 0.25,
        "gsde_init_stds": [1.0, 1.0],
        "mat_add_agent_embeddings": False,
        "mat_use_agent_attention": True,
        "mat_decoder_self_attention_mode": MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        "act_fn_cls": nn.GELU,
        "mat_init_gains": MATInitGains(),
        "mat_normalization": mat_normalization,
        "assume_agent_mask_is_active_prefix": False,
    }
    kwargs.update(overrides)
    return _make_base_policy(**kwargs)


def test_make_vector_env_forwards_tensor_compile_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = object()
    vector_env = object()
    constructor = Mock(return_value=vector_env)
    monkeypatch.setattr(experiment_common, "_make_scenario", Mock(return_value=scenario))
    monkeypatch.setattr(experiment_common, "MJWSwarmBotsVectorEnv", constructor)

    result = experiment_common.make_vector_env(
        episode_length=32,
        num_envs=4,
        first_episode_lengths=[8, 16, 24, 32],
        settle_initial_reset=True,
        device=torch.device("cuda"),
        compile_env_tensor_operations=False,
        env_tensor_operations_compile_mode="reduce-overhead",
    )

    assert result is vector_env
    constructor.assert_called_once_with(
        scenario=scenario,
        num_envs=4,
        episode_length=32,
        first_episode_lengths=[8, 16, 24, 32],
        settle_initial_reset=True,
        device=torch.device("cuda"),
        ccd_iterations=None,
        compile_tensor_operations=False,
        tensor_operations_compile_mode="reduce-overhead",
    )


def test_default_run_experiment_wires_ppo_contract(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    capture, env = _patch_default_experiment_boundaries(monkeypatch)
    entrypoint_path = tmp_path / "experiment_entrypoint.py"
    entrypoint_path.write_text("# test experiment\n", encoding="utf-8")

    experiment_common.run_experiment(
        num_envs=4,
        rollout_steps_per_env=2,
        variant_name="default-contract",
        entrypoint_path=entrypoint_path,
        total_timesteps=16,
    )

    vector_env_kwargs = capture["vector_env_kwargs"]
    assert isinstance(vector_env_kwargs, dict)
    assert vector_env_kwargs["episode_length"] == 512
    assert vector_env_kwargs["first_episode_lengths"] == [128, 256, 384, 512]
    assert vector_env_kwargs["settle_initial_reset"] is True
    assert vector_env_kwargs["scenario_name"] == "wall"
    assert vector_env_kwargs["compile_env_tensor_operations"] is None
    assert vector_env_kwargs["env_tensor_operations_compile_mode"] == "default"
    assert vector_env_kwargs["scenario_kwargs"] == {"continuous_connector_actions": True}

    wrap_vec_env_kwargs = capture["wrap_vec_env_kwargs"]
    assert isinstance(wrap_vec_env_kwargs, dict)
    assert wrap_vec_env_kwargs["use_popart"] is True
    assert wrap_vec_env_kwargs["use_transition_obs"] is False
    assert wrap_vec_env_kwargs["shuffle_agents"] is False

    base_policy_kwargs = capture["base_policy_kwargs"]
    assert isinstance(base_policy_kwargs, dict)
    assert base_policy_kwargs["policy_variant"] == "mat_qcs"
    assert base_policy_kwargs["continuous_action_dist"] == "sign_magnitude_beta"
    assert base_policy_kwargs["use_nop"] is True
    assert base_policy_kwargs["world_model_num_next_steps"] == 3
    assert base_policy_kwargs["assume_agent_mask_is_active_prefix"] is True

    world_model_config = capture["world_model_config"]
    assert world_model_config.n_agents == 3
    assert world_model_config.local_latent_dim == 8
    assert world_model_config.action_dim == 3
    assert world_model_config.world_model_loss_coef == 0.1
    assert world_model_config.next_obs_pred_config.local_scalar_target_indices == [0, 1]

    ppo_kwargs = capture["ppo_kwargs"]
    assert isinstance(ppo_kwargs, dict)
    assert ppo_kwargs["rollout_mode"].n_steps_per_rollout == 8
    assert ppo_kwargs["sampler_config"].batch_size == 8
    assert ppo_kwargs["sampler_config"].num_next_steps == 3
    assert ppo_kwargs["rollout_warmup_steps_per_env"] == 512
    assert ppo_kwargs["use_popart"] is True
    assert ppo_kwargs["learning_rate"].initial_lr == pytest.approx(5e-7)
    assert ppo_kwargs["learning_rate"].max_lr == pytest.approx(8e-4)
    assert ppo_kwargs["clip_range"] == pytest.approx(0.05)
    assert ppo_kwargs["target_kl"] == pytest.approx(0.002)
    assert set(ppo_kwargs["parameter_lr_multipliers"].values()) == {0.25}
    assert "decoder" in ppo_kwargs["parameter_lr_multipliers"]
    assert "actor_head" not in ppo_kwargs["parameter_lr_multipliers"]

    learn_kwargs = capture["learn_kwargs"]
    assert isinstance(learn_kwargs, dict)
    assert learn_kwargs["max_total_timesteps"] == 16
    assert learn_kwargs["save_interval"] is None
    assert learn_kwargs["save_optimizer"] is True
    assert learn_kwargs["best_rotation_n"] == 1
    assert learn_kwargs["log_interval"] == 1
    assert learn_kwargs["logging_buffer_size"] == 5
    metadata = learn_kwargs["extra_run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["algorithm_variant"] == "ppo"
    assert metadata["policy_variant"] == "mat_qcs"
    assert metadata["rollout_samples"] == 8
    assert metadata["total_timesteps"] == 16
    assert metadata["logging_buffer_size"] == 5
    assert metadata["world_model_num_next_steps"] == 3
    assert metadata["scenario_kwargs"] == {"continuous_connector_actions": True}
    assert metadata["bernoulli_initial_prob"] == 0.8
    assert metadata["enc_nhead"] == 4
    assert metadata["dec_nhead"] == 2
    assert metadata["mat_decoder_lr_multiplier"] == pytest.approx(0.25)
    assert metadata["include_actor_head_lr_multiplier"] is False
    assert metadata["parameter_lr_multipliers"] == ppo_kwargs["parameter_lr_multipliers"]
    assert learn_kwargs["post_iteration_hooks"] == [capture["recording_hook"]]
    recording_kwargs = capture["recording_kwargs"]
    assert isinstance(recording_kwargs, dict)
    assert recording_kwargs == {
        "algorithm": capture["algorithm"],
        "total_timesteps": 16,
        "schedule": experiment_common.DEFAULT_LIVE_RECORDING_SCHEDULE,
    }
    logging_key_names = {entry[0] for entry in learn_kwargs["logging_console_keys"]}
    assert "act0_j0" in logging_key_names
    assert "act0_j1" in logging_key_names
    assert "std0_j0" not in logging_key_names
    assert "std0_j1" not in logging_key_names
    assert env.closed


def test_tmasac_run_experiment_preserves_sac_and_nop_configuration(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, env = _patch_default_experiment_boundaries(monkeypatch)
    transformer_ff_config = SwiGLUConfig(
        hidden_dim=12,
        stacked=GLUStackConfig(
            n_layers=2,
            pre_norm=nn.LayerNorm,
            norm_first_layer=False,
            residual=True,
            residual_first_layer=False,
        ),
    )

    experiment_common.run_experiment(
        num_envs=4,
        rollout_steps_per_env=2,
        variant_name="tmasac-contract",
        entrypoint_path=Path(__file__),
        policy_variant="tmasac",
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
        nop_skip_first_transition_for_critic=False,
        sac_ent_coef="auto_0.2",
        sac_target_entropy="auto_0.7",
        sac_independent_nop_sampling=True,
        mat_encoder_transformer_ff_config=transformer_ff_config,
        compile_env_tensor_operations=False,
        env_tensor_operations_compile_mode="reduce-overhead",
        total_timesteps=16,
    )

    vector_env_kwargs = capture["vector_env_kwargs"]
    assert isinstance(vector_env_kwargs, dict)
    assert vector_env_kwargs["compile_env_tensor_operations"] is False
    assert vector_env_kwargs["env_tensor_operations_compile_mode"] == "reduce-overhead"

    wrap_vec_env_kwargs = capture["wrap_vec_env_kwargs"]
    assert isinstance(wrap_vec_env_kwargs, dict)
    assert wrap_vec_env_kwargs["use_popart"] is False

    base_policy_kwargs = capture["base_policy_kwargs"]
    assert isinstance(base_policy_kwargs, dict)
    assert base_policy_kwargs["policy_variant"] == "tmasac"
    assert (
        base_policy_kwargs["continuous_action_dist"]
        == "reparameterized_sign_magnitude_kumaraswamy"
    )
    assert base_policy_kwargs["use_popart"] is False
    assert base_policy_kwargs["world_model_num_next_steps"] == 3
    assert base_policy_kwargs["nop_skip_first_transition_for_critic"] is False
    assert "world_model_config" not in capture

    sac_kwargs = capture["sac_kwargs"]
    assert isinstance(sac_kwargs, dict)
    assert sac_kwargs["learning_rate"] == 3e-4
    assert sac_kwargs["rollout_steps_per_iteration"] == 8
    assert sac_kwargs["rollout_warmup_steps_per_env"] == 512
    assert sac_kwargs["buffer_capacity_per_env"] * 4 >= max(
        sac_kwargs["learning_starts"],
        sac_kwargs["batch_size"],
    )
    assert sac_kwargs["ent_coef"] == "auto_0.2"
    assert sac_kwargs["ent_coef_learning_rate"] == 1e-3
    assert sac_kwargs["target_entropy"] == "auto_0.7"
    assert sac_kwargs["independent_nop_sampling"] is True
    assert sac_kwargs["gsde_reset_mode"].probability == pytest.approx(1 / 6)

    metrics_action_splitters = sac_kwargs["metrics_action_splitters"]
    actuator_metrics = metrics_action_splitters[0](torch.arange(8).reshape(1, 8))
    torch.testing.assert_close(actuator_metrics["j0"], torch.tensor([[0, 2, 4, 6]]))
    torch.testing.assert_close(actuator_metrics["j1"], torch.tensor([[1, 3, 5, 7]]))

    learn_kwargs = capture["learn_kwargs"]
    assert isinstance(learn_kwargs, dict)
    assert learn_kwargs["log_interval"] == 1
    assert learn_kwargs["logging_buffer_size"] == 20
    metadata = learn_kwargs["extra_run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["algorithm_variant"] == "sac"
    assert metadata["policy_variant"] == "tmasac"
    assert metadata["nop_skip_first_transition_for_critic"] is False
    assert metadata["sac_independent_nop_sampling"] is True
    assert metadata["sac_ent_coef"] == "auto_0.2"
    assert metadata["sac_ent_coef_learning_rate"] == 1e-3
    assert metadata["sac_target_entropy"] == "auto_0.7"
    assert metadata["logging_buffer_size"] == 20
    assert metadata["mat_encoder_transformer_ff_config"] == {
        "hidden_dim": 12,
        "stacked": {
            "n_layers": 2,
            "pre_norm": "torch.nn.modules.normalization.LayerNorm",
            "norm_first_layer": False,
            "residual": True,
            "residual_first_layer": False,
        },
    }
    assert "sac_nop_steps" not in metadata
    logging_key_names = {entry[0] for entry in learn_kwargs["logging_console_keys"]}
    assert "actor_action_dist_act0_entropy_loss_scaled" not in logging_key_names
    assert "actor_action_dist_act1_entropy_loss_scaled" not in logging_key_names
    assert "actor_action_dist_act0_ent_categorical" not in logging_key_names
    assert "actor_action_dist_act0_ent_kumaraswamy" not in logging_key_names
    assert env.closed


def test_recurrent_tmasac_run_experiment_wires_recurrent_replay_and_actor_layout(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    capture, env = _patch_default_experiment_boundaries(monkeypatch)
    entrypoint_path = tmp_path / "recurrent_tmasac_entrypoint.py"
    entrypoint_path.write_text("# test recurrent TMASAC experiment\n", encoding="utf-8")

    experiment_common.run_experiment(
        num_envs=4,
        rollout_steps_per_env=1,
        variant_name="recurrent-tmasac-contract",
        entrypoint_path=entrypoint_path,
        policy_variant="r_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        rmat_actor_d_model=64,
        rmat_actor_transformer_ff_config=MLPConfig(hidden_dims=[128]),
        rmat_actor_inter_module_mlp=True,
        rmat_experimental_compile_lstm=True,
        tmasac_separate_observation_action_encoders=True,
        sac_batch_size=16,
        sac_buffer_capacity_per_env=160,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=32,
        sac_temporal_state_storage_dtype=torch.float16,
        total_timesteps=16,
    )

    base_policy_kwargs = capture["base_policy_kwargs"]
    assert isinstance(base_policy_kwargs, dict)
    assert base_policy_kwargs["policy_variant"] == "r_tmasac"
    assert base_policy_kwargs["compile_policy_modules"] is True
    assert base_policy_kwargs["compile_world_model_modules"] is True
    assert base_policy_kwargs["rmat_experimental_compile_lstm"] is True
    assert base_policy_kwargs["rmat_actor_d_model"] == 64
    assert base_policy_kwargs["rmat_actor_transformer_ff_config"] == MLPConfig(hidden_dims=[128])
    assert base_policy_kwargs["rmat_actor_inter_module_mlp"] is True
    assert base_policy_kwargs["tmasac_separate_observation_action_encoders"] is True

    recurrent_sac_kwargs = capture["recurrent_sac_kwargs"]
    assert isinstance(recurrent_sac_kwargs, dict)
    assert recurrent_sac_kwargs["buffer_capacity_per_env"] == 160
    assert recurrent_sac_kwargs["batch_size"] == 16
    assert recurrent_sac_kwargs["burn_in_steps"] == 32
    assert recurrent_sac_kwargs["learning_steps"] == 64
    assert recurrent_sac_kwargs["temporal_state_store_interval"] == 32
    assert recurrent_sac_kwargs["temporal_state_storage_dtype"] is torch.float16
    assert recurrent_sac_kwargs["max_truncations_per_segment"] == 1
    assert recurrent_sac_kwargs["independent_nop_sampling"] is False

    learn_kwargs = capture["learn_kwargs"]
    assert isinstance(learn_kwargs, dict)
    metadata = learn_kwargs["extra_run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["recurrent_policy"] is True
    assert metadata["sac_buffer_capacity_per_env"] == 160
    assert metadata["sac_batch_size"] == 16
    assert metadata["sac_temporal_state_storage_dtype"] == "torch.float16"
    assert env.closed


def test_segment_tmasac_run_experiment_wires_recurrent_segment_sampling(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    capture, env = _patch_default_experiment_boundaries(monkeypatch)
    entrypoint_path = tmp_path / "segment_tmasac_entrypoint.py"
    entrypoint_path.write_text("# test segment TMASAC experiment\n", encoding="utf-8")

    experiment_common.run_experiment(
        num_envs=4,
        rollout_steps_per_env=1,
        variant_name="segment-tmasac-contract",
        entrypoint_path=entrypoint_path,
        policy_variant="segment_tmasac",
        sac_batch_size=16,
        sac_buffer_capacity_per_env=160,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=16,
        total_timesteps=16,
    )

    base_policy_kwargs = capture["base_policy_kwargs"]
    assert isinstance(base_policy_kwargs, dict)
    assert base_policy_kwargs["policy_variant"] == "segment_tmasac"

    recurrent_sac_kwargs = capture["recurrent_sac_kwargs"]
    assert isinstance(recurrent_sac_kwargs, dict)
    assert recurrent_sac_kwargs["batch_size"] == 16
    assert recurrent_sac_kwargs["burn_in_steps"] == 32
    assert recurrent_sac_kwargs["learning_steps"] == 64
    assert recurrent_sac_kwargs["temporal_state_store_interval"] == 16

    learn_kwargs = capture["learn_kwargs"]
    assert isinstance(learn_kwargs, dict)
    metadata = learn_kwargs["extra_run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["algorithm_variant"] == "sac"
    assert metadata["policy_variant"] == "segment_tmasac"
    assert metadata["sac_recurrent_learning_steps"] == 64
    assert env.closed


@pytest.mark.parametrize(
    ("num_envs", "rollout_steps_per_env", "policy_variant", "expected_message"),
    [
        (3, 1, "mat_qcs", "rollout_samples divisible"),
        (3, 2, "r_mat_qcs", "sampler_batch_size divisible"),
    ],
)
def test_run_experiment_rejects_virtual_batches_that_split_sampling_units(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        num_envs: int,
        rollout_steps_per_env: int,
        policy_variant: PolicyVariant,
        expected_message: str,
) -> None:
    capture, _env = _patch_default_experiment_boundaries(monkeypatch)

    with pytest.raises(ValueError, match=expected_message):
        experiment_common.run_experiment(
            num_envs=num_envs,
            rollout_steps_per_env=rollout_steps_per_env,
            virtual_mini_batches=2,
            variant_name="invalid-virtual-batches",
            entrypoint_path=tmp_path / "unused.py",
            policy_variant=policy_variant,
            total_timesteps=16,
        )

    assert "vector_env_kwargs" not in capture


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"num_envs": 0}, "num_envs must be > 0"),
        ({"rollout_steps_per_env": 0}, "rollout_steps_per_env must be > 0"),
        ({"virtual_mini_batches": 0}, "virtual_mini_batches must be > 0"),
        ({"n_epochs": 0}, "n_epochs must be > 0"),
        ({"total_timesteps": 0}, "total_timesteps must be > 0"),
        ({"sac_batch_size": 0}, "sac_batch_size must be > 0"),
    ],
)
def test_run_experiment_rejects_non_positive_sampling_dimensions(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        overrides: dict[str, int],
        expected_message: str,
) -> None:
    capture, _env = _patch_default_experiment_boundaries(monkeypatch)
    kwargs: dict[str, object] = {
        "num_envs": 4,
        "rollout_steps_per_env": 2,
        "virtual_mini_batches": 1,
        "variant_name": "invalid-sampling-dimensions",
        "entrypoint_path": tmp_path / "unused.py",
        "total_timesteps": 16,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=expected_message):
        experiment_common.run_experiment(**kwargs)

    assert "vector_env_kwargs" not in capture


def test_mat_qcs_and_qcc_lr_multipliers_do_not_include_qcx_action_modules() -> None:
    expected_prefixes = {
        "decoder",
        "query_input_norm",
        "query_encoder",
        "query_token_norm",
        "context_input_norm",
        "context_encoder",
        "context_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "agent_embeddings_decoder",
    }

    for policy_variant in ("mat_qcs", "mat_qcc"):
        multipliers = _make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            mat_decoder_lr_multiplier=0.25,
        )

        assert set(multipliers) == expected_prefixes
        assert set(multipliers.values()) == {0.25}


def test_mat_qcx_lr_multipliers_include_decoder_and_memory_modules() -> None:
    multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=0.25,
    )

    assert set(multipliers) == {
        "decoder",
        "action_input_norm",
        "action_encoder",
        "action_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
    }
    assert set(multipliers.values()) == {0.25}


def test_mat_lr_multipliers_can_include_actor_head_modules() -> None:
    expected_qcs_qcc_prefixes = {
        "decoder",
        "query_input_norm",
        "query_encoder",
        "query_token_norm",
        "context_input_norm",
        "context_encoder",
        "context_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "agent_embeddings_decoder",
        "actor_head_input_norm",
        "actor_head",
    }

    for policy_variant in ("mat_qcs", "mat_qcc"):
        multipliers = _make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            mat_decoder_lr_multiplier=0.25,
            include_actor_head_lr_multiplier=True,
        )

        assert set(multipliers) == expected_qcs_qcc_prefixes
        assert set(multipliers.values()) == {0.25}

    qcx_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=0.25,
        include_actor_head_lr_multiplier=True,
    )

    assert set(qcx_multipliers) == {
        "decoder",
        "action_input_norm",
        "action_encoder",
        "action_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "actor_head_input_norm",
        "actor_head",
    }
    assert set(qcx_multipliers.values()) == {0.25}

    mat_ind_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_ind",
        mat_decoder_lr_multiplier=0.25,
        include_actor_head_lr_multiplier=True,
    )

    assert mat_ind_multipliers == {"actor_head": 0.25}

    mat_dec_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_dec",
        mat_decoder_lr_multiplier=0.25,
        include_actor_head_lr_multiplier=True,
    )

    assert mat_dec_multipliers == {"actor_head": 0.25}


def test_make_base_policy_constructs_mat_dec_variant() -> None:
    policy = _make_test_base_policy(policy_variant="mat_dec")

    assert isinstance(policy, MATDecPolicy)
    assert policy.actor_head.input_dim == policy.actor_encoder_config.d_model
    assert policy.actor_head.hidden_dims == [8, 8]
    assert policy.actor_encoder.global_obs_dim == _DummyEnv.global_obs_dim
    assert all(layer.self_attn is None for layer in policy.actor_encoder.layers)


def test_make_base_policy_can_disable_mat_ind_agent_attention() -> None:
    policy = _make_test_base_policy(
        policy_variant="mat_ind",
        mat_use_agent_attention=False,
    )

    assert isinstance(policy.encoder, MATEncoder)
    assert all(layer.self_attn is None for layer in policy.encoder.layers)


def test_make_base_policy_constructs_mat_qcx_variant() -> None:
    policy = _make_test_base_policy(
        policy_variant="mat_qcx",
        mat_normalization=MATNormalizationConfig(
            normalize_action_input=True,
            normalize_action_tokens=True,
        ),
    )

    assert isinstance(policy, MATQCXPolicy)
    assert isinstance(policy.action_input_norm, nn.LayerNorm)
    assert isinstance(policy.action_token_norm, nn.LayerNorm)
    assert isinstance(policy.decoder.layers[0].context_token_norm, nn.LayerNorm)


@pytest.mark.parametrize(
    ("actor_head_kind", "expected_qcx", "expected_agent_attention"),
    [
        (TMASACActorHeadKind.INDEPENDENT, False, True),
        (TMASACActorHeadKind.QCX, True, True),
        (TMASACActorHeadKind.DECENTRALIZED, False, False),
    ],
)
def test_make_base_policy_constructs_tmasac_actor_variants(
        actor_head_kind: TMASACActorHeadKind,
        expected_qcx: bool,
        expected_agent_attention: bool,
) -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="tmasac",
        continuous_action_dist="predicted_std",
        tmasac_actor_head_kind=actor_head_kind,
    )

    assert isinstance(policy, TMASACPolicy)
    assert isinstance(policy.actor_head, TMASACQCXActorHead) is expected_qcx
    assert all(
        (layer.self_attn is not None) is expected_agent_attention
        for layer in policy.actor_encoder.layers
    )


def test_default_base_policy_produces_finite_actions_log_probs_and_values() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="mat_qcs",
        use_popart=True,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
    )
    batch_size = 2

    assert isinstance(policy, MATQCSPolicy)
    assert policy.has_popart
    assert all(isinstance(dist, SignMagnitudeBetaActionDist) for dist in policy.action_dist.distributions)
    continuous_dist = policy.action_dist.distributions[0]
    assert continuous_dist.ent_loss_coef == pytest.approx(1e-3)
    assert continuous_dist.beta_ent_scale == pytest.approx(0.75)
    assert continuous_dist.categorical_ent_loss_config.entropy_floor == 0.35
    assert continuous_dist.beta_ent_loss_config.entropy_floor is None

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=torch.randn(batch_size, _DummyEnv.n_agents, _DummyEnv.local_obs_dim),
            global_obs=torch.randn(batch_size, _DummyEnv.global_obs_dim),
            hidden_local_vars=torch.empty(batch_size, _DummyEnv.n_agents, 0),
            hidden_global_vars=torch.empty(batch_size, 0),
            agent_mask=torch.ones(batch_size, _DummyEnv.n_agents, dtype=torch.bool),
            deterministic=True,
        )

    assert actions.shape == (batch_size, _DummyEnv.n_agents, 3)
    assert log_probs.shape == (batch_size, _DummyEnv.n_agents)
    assert values.shape == (batch_size,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_binary_connector_policy_starts_from_requested_connection_probability() -> None:
    policy = _make_test_base_policy(
        policy_variant="mat_qcs",
        bernoulli_initial_prob=0.3,
    )
    connector_dist = policy.action_dist.distributions[1]

    assert isinstance(connector_dist, BernoulliActionDist)

    connector_dist.update_latent_features(
        torch.zeros(2, _DummyEnv.n_agents, connector_dist.latent_dim)
    )

    assert connector_dist.distribution is not None
    torch.testing.assert_close(
        connector_dist.distribution.probs,
        torch.full_like(connector_dist.distribution.probs, 0.3),
    )


def test_make_base_policy_passes_rmat_temporal_output_projection_flag() -> None:
    policy = _make_test_base_policy(
        policy_variant="r_mat_ind",
        rmat_use_temporal_output_projection=False,
    )

    assert isinstance(policy, RMATIndPolicy)
    assert all(isinstance(layer.temporal_output_projection, nn.Identity) for layer in policy.encoder.layers)


def test_first_episode_staggering_never_creates_zero_length_episodes() -> None:
    lengths = _make_staggered_first_episode_lengths(
        episode_length=512,
        num_envs=1024,
    )

    assert lengths[0] == 1
    assert lengths[-1] == 512
    assert all(length >= 1 for length in lengths)
    assert all(lengths[idx] <= lengths[idx + 1] for idx in range(len(lengths) - 1))
    assert all(lengths.count(length) == 2 for length in range(1, 513))


def test_default_scenario_uses_continuous_connectors_without_overriding_explicit_choice() -> None:
    assert _with_default_scenario_kwargs(None) == {"continuous_connector_actions": True}
    assert _with_default_scenario_kwargs({"continuous_connector_actions": False}) == {
        "continuous_connector_actions": False,
    }
    supplied_kwargs = {"difficulty": "medium"}

    resolved_kwargs = _with_default_scenario_kwargs(supplied_kwargs)

    assert resolved_kwargs == {"continuous_connector_actions": True, "difficulty": "medium"}
    assert supplied_kwargs == {"difficulty": "medium"}


def test_default_ppo_wrapper_stack_preserves_same_step_processing_order() -> None:
    env = wrap_vec_env(
        vector_env=_make_wrapper_test_vector_env(),
        obs_indices=_make_obs_indices(),
        gamma=0.99,
        use_popart=True,
        rollout_device=torch.device("cpu"),
    )

    try:
        wrappers: list[object] = []
        current_env = env
        while hasattr(current_env, "env"):
            wrappers.append(current_env)
            current_env = current_env.env

        assert [wrapper.obs_key for wrapper in wrappers[:4]] == [
            "hidden_global_vars",
            "hidden_local_vars",
            "global_obs",
            "local_obs",
        ]
        assert all(isinstance(wrapper, TorchFeatureWiseObsNormWrapper) for wrapper in wrappers[:4])
        assert isinstance(wrappers[4], TorchProgressGuidanceEpisodeStatsWrapper)
        assert isinstance(wrappers[5], TorchRecordEpisodeStatisticsWrapper)
        assert isinstance(wrappers[6], SwarmBotsLearnEnvWrapper)
    finally:
        env.close()


def test_sac_wrapper_normalizes_rewards_outside_observation_processing() -> None:
    env = wrap_vec_env(
        vector_env=_make_wrapper_test_vector_env(),
        obs_indices=_make_obs_indices(),
        gamma=0.99,
        use_popart=False,
        rollout_device=torch.device("cpu"),
    )

    try:
        assert isinstance(env, TorchNormalizeRewardWrapper)
        assert isinstance(env.env, TorchFeatureWiseObsNormWrapper)
        assert env.env.obs_key == "hidden_global_vars"
    finally:
        env.close()


def test_training_failure_still_closes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Mock()
    expected_error = RuntimeError("training failed")

    def run_with_error() -> None:
        raise expected_error

    def run_notification(*, run: Callable[[], None], **_kwargs: object) -> None:
        run()

    monkeypatch.setattr(
        "experiments.mjw_experiment_common.run_with_discord_notification",
        run_notification,
    )

    with pytest.raises(RuntimeError, match="training failed") as error_info:
        _run_training_with_notification_and_close(
            env=env,
            run_name="test/run",
            run_dir="runs/test",
            total_timesteps=10,
            algorithm=Mock(),
            run=run_with_error,
        )

    assert error_info.value is expected_error
    env.close.assert_called_once_with()


def test_non_gsde_policy_ignores_irrelevant_joint_std_layout() -> None:
    policy = _make_test_base_policy(policy_variant="mat_qcs")

    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=3,
        joint_stds=[0.25, 0.30],
    )


def test_gsde_joint_stds_are_assigned_by_joint_across_limbs() -> None:
    policy = _make_test_base_policy(
        policy_variant="mat_qcs",
        continuous_action_dist="gsde",
    )
    dist = policy.action_dist.distributions[0]

    assert isinstance(dist, GSDEActionDist)

    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=2,
        joint_stds=[0.2, 0.4],
    )

    torch.testing.assert_close(dist.log_stds[:, 0::2].exp(), torch.full_like(dist.log_stds[:, 0::2], 0.2))
    torch.testing.assert_close(dist.log_stds[:, 1::2].exp(), torch.full_like(dist.log_stds[:, 1::2], 0.4))


def test_make_base_policy_constructs_tmasac_rsmk_with_nop() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="tmasac",
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
        use_nop=True,
        obs_indices=_make_obs_indices(),
    )

    assert isinstance(policy, TMASACPolicy)
    assert isinstance(policy.action_dist.distributions[0], ReparameterizedSignMagnitudeKumaraswamyActionDist)
    assert isinstance(policy.action_dist.distributions[1], ReparameterizedSignMagnitudeKumaraswamyActionDist)
    assert policy.action_dist.distributions[0].ent_loss_coef == 0.0
    assert policy.action_dist.distributions[1].ent_loss_coef == 0.0
    assert policy.action_dist.distributions[0].kumaraswamy_ent_scale == 0.0
    assert policy.action_dist.distributions[1].kumaraswamy_ent_scale == 0.0
    assert policy.action_dist.distributions[0].categorical_ent_loss_config.entropy_floor == 0.35
    assert policy.action_dist.distributions[0].kumaraswamy_ent_loss_config.entropy_floor is None
    assert policy.actor_nop is None
    assert policy.critic_nop is not None
    assert policy.critic_nop.skip_first_transition
    assert policy.critic_nop.latent_projection_hidden_dims == [8, 8]
    assert policy.critic_nop.local_scalar_target_indices == [0, 1]


def test_make_base_policy_constructs_recurrent_tmasac_with_feedforward_critic() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="r_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        rmat_actor_d_model=16,
        rmat_actor_transformer_ff_config=MLPConfig(hidden_dims=[32]),
        rmat_actor_inter_module_mlp=True,
    )

    assert isinstance(policy, RecurrentTMASACPolicy)
    assert policy.recurrent_critic is False
    assert policy._actor_encoder.d_model == 16
    assert policy.critic.d_model == 8
    for layer in policy._actor_encoder.layers:
        inter_module_mlp = layer.inter_module_feedforward
        assert inter_module_mlp is not None
        final_projection = inter_module_mlp[-1]
        assert isinstance(final_projection, nn.Linear)


def test_make_base_policy_accepts_swiglu_configs_for_mat_and_recurrent_mat() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="r_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        mat_encoder_transformer_ff_config=SwiGLUConfig(
            hidden_dim=12,
            stacked=GLUStackConfig(n_layers=2),
        ),
        rmat_actor_d_model=8,
        rmat_actor_transformer_ff_config=SwiGLUConfig(
            hidden_dim=10,
            stacked=GLUStackConfig(n_layers=3),
        ),
    )

    assert isinstance(policy, RecurrentTMASACPolicy)
    assert all(
        isinstance(layer.feedforward, StackedGLU) and len(layer.feedforward) == 3
        for layer in policy._actor_encoder.layers
    )
    assert all(
        isinstance(layer.feedforward, StackedGLU) and len(layer.feedforward) == 2
        for layer in policy.critic.encoder.layers
    )


def test_experiment_entrypoint_exposes_only_typed_feedforward_configs() -> None:
    parameters = inspect.signature(experiment_common.run_experiment).parameters

    assert "mat_encoder_transformer_ff_config" in parameters
    assert "rmat_actor_transformer_ff_config" in parameters
    assert "mat_encoder_transformer_ff_hidden_dims" not in parameters
    assert "rmat_actor_transformer_ff_hidden_dims" not in parameters


def test_find_opening_recurrent_feedforward_configs_reach_critic_and_actor(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    swiglu_config = SwiGLUConfig(
        hidden_dim=find_opening_common.PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM
    )
    monkeypatch.setattr(
        find_opening_common,
        "run_mjw_find_opening_experiment",
        lambda **kwargs: captured_kwargs.update(kwargs),
    )

    find_opening_common.run_experiment(
        variant_name="test",
        entrypoint_path=Path(__file__),
        temporal_model_variant="lstm",
        mat_encoder_transformer_ff_config=swiglu_config,
        rmat_actor_transformer_ff_config=swiglu_config,
    )

    assert captured_kwargs["mat_encoder_transformer_ff_config"] is swiglu_config
    assert captured_kwargs["rmat_actor_transformer_ff_config"] is swiglu_config


def test_make_base_policy_enables_separate_tmasac_observation_action_encoders() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="r_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        tmasac_separate_observation_action_encoders=True,
    )

    assert isinstance(policy, RecurrentTMASACPolicy)
    assert policy.config.critic_config.separate_observation_action_encoders
    assert policy.critic.encoder.observation_action_encoder is not None


def test_make_base_policy_wires_recurrent_actor_state_critic_input() -> None:
    actor_state_config = ActorStateCriticInputConfig(
        projection_dim=6,
        projection_hidden_dims=(10,),
    )
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="r_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        r_tmasac_actor_state_critic_input_config=actor_state_config,
    )

    assert isinstance(policy, RecurrentTMASACPolicy)
    assert policy.config.actor_state_critic_input_config == actor_state_config
    assert policy.uses_actor_state_critic_input


def test_make_base_policy_constructs_segment_tmasac_with_regular_mat() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="segment_tmasac",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
    )

    assert isinstance(policy, SegmentTMASACPolicy)
    assert isinstance(policy.actor_encoder, MATEncoder)
    assert policy.recurrent_critic is False
    assert not policy.uses_temporal_actor_state


def test_big_end_and_two_small_recurrent_actor_mlps_have_similar_parameter_counts() -> None:
    common_kwargs = {
        "env": _DummyContinuousEnv(),
        "policy_variant": "r_tmasac",
        "continuous_action_dist": "gumbel_softmax_sign_magnitude_beta",
        "rmat_actor_d_model": 8,
    }
    big_end_policy = _make_test_base_policy(
        **common_kwargs,
        rmat_actor_transformer_ff_config=MLPConfig(hidden_dims=[16, 16]),
    )
    two_small_policy = _make_test_base_policy(
        **common_kwargs,
        rmat_actor_transformer_ff_config=MLPConfig(hidden_dims=[16]),
        rmat_actor_inter_module_mlp=True,
    )

    def count_mlp_parameters(policy: RecurrentTMASACPolicy) -> int:
        return sum(
            parameter.numel()
            for layer in policy._actor_encoder.layers
            for mlp in (layer.feedforward, layer.inter_module_feedforward)
            if mlp is not None
            for parameter in mlp.parameters()
        )

    assert count_mlp_parameters(two_small_policy) == pytest.approx(
        count_mlp_parameters(big_end_policy),
        rel=0.02,
    )
    big_end_encoder_parameters = sum(parameter.numel() for parameter in big_end_policy._actor_encoder.parameters())
    two_small_encoder_parameters = sum(
        parameter.numel() for parameter in two_small_policy._actor_encoder.parameters()
    )
    assert two_small_encoder_parameters == pytest.approx(big_end_encoder_parameters, rel=0.02)


def test_tmasac_nop_requires_observation_target_indices() -> None:
    with pytest.raises(ValueError, match="obs_indices is required"):
        _make_test_base_policy(
            env=_DummyContinuousEnv(),
            policy_variant="tmasac",
            continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
            use_nop=True,
        )


def test_make_base_policy_can_keep_first_critic_nop_transition() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="tmasac",
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
        use_nop=True,
        nop_skip_first_transition_for_critic=False,
        obs_indices=_make_obs_indices(),
    )

    assert isinstance(policy, TMASACPolicy)
    assert policy.critic_nop is not None
    assert not policy.critic_nop.skip_first_transition


def test_make_base_policy_preserves_requested_tmasac_nop_horizon() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="tmasac",
        continuous_action_dist="reparameterized_sign_magnitude_kumaraswamy",
        use_nop=True,
        world_model_num_next_steps=7,
        obs_indices=_make_obs_indices(),
    )

    assert isinstance(policy, TMASACPolicy)
    assert policy.has_nop_loss()
    assert policy.get_nop_num_next_steps() == 7


def test_make_base_policy_supports_tmasac_straight_through_action_families() -> None:
    variants_and_types = (
        ("gumbel_softmax_sign_magnitude_beta", GumbelSoftmaxSignMagnitudeBetaActionDist),
        ("ternary_sign_magnitude_beta", TernarySignMagnitudeBetaActionDist),
        (
            "gumbel_softmax_sign_magnitude_kumaraswamy",
            GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
        ),
    )

    for variant, expected_type in variants_and_types:
        policy = _make_test_base_policy(
            env=_DummyContinuousEnv(),
            policy_variant="tmasac",
            continuous_action_dist=variant,
        )

        assert isinstance(policy, TMASACPolicy)
        assert all(isinstance(dist, expected_type) for dist in policy.action_dist.distributions)
        for dist in policy.action_dist.distributions:
            assert dist.ent_loss_coef == 0.0
            assert dist.compute_extra_losses() == ({}, {})

    kumaraswamy_dist = policy.action_dist.distributions[0]
    assert isinstance(kumaraswamy_dist, GumbelSoftmaxSignMagnitudeKumaraswamyActionDist)
    assert kumaraswamy_dist.kumaraswamy_ent_scale == 0.0
    assert kumaraswamy_dist.categorical_ent_loss_config.entropy_floor == 0.35


def test_make_base_policy_uses_action_gain_for_predicted_std_log_std_head() -> None:
    policy = _make_test_base_policy(
        env=_DummyContinuousEnv(),
        policy_variant="tmasac",
        continuous_action_dist="predicted_std",
        mat_init_gains=MATInitGains(action_net=0.0),
        gsde_init_stds=[0.25, 0.30],
    )
    dist = policy.action_dist.distributions[0]

    assert isinstance(dist, PredictedStdActionDist)
    latent = torch.randn(4, _DummyContinuousEnv.n_agents, policy.actor_head.latent_dim)
    dist.update_latent_features(latent)

    assert torch.allclose(dist.log_std_net.weight, torch.zeros_like(dist.log_std_net.weight))
    assert torch.allclose(dist.distribution.scale, torch.full_like(dist.distribution.scale, 0.25))


def test_mat_lr_multiplier_one_disables_parameter_overrides() -> None:
    multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=1.0,
    )

    assert multipliers == {}

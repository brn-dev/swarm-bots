from collections.abc import Iterator

import pytest
import torch

from swarmbots import make_env, make_scenario
from swarmbots.learn.presets.nop import make_next_obs_pred_config
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sequence_runtime import _advance_goal_sequence
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig

TASK = "SwarmBots-PayloadGoalSequence-v0"


def _scenario_kwargs() -> dict[str, object]:
    return {
        "unit_start_locations":MJWPreConnectedUnitLocationsConfig(2,pool_seeds=(42000,)),
        "compile_reward_kernel":False,"reset_settle_time":0.,"goal_hold_time":.06,
        "include_goal_delta_obs":True,"include_payload_velocity_obs":True,
        "include_goal_inside_obs":True,
        "goal_hold_reward_fraction":.5,"towards_payload_potential":"pseudo_huber",
    }


@pytest.fixture(params=[0., .06])
def motion_env(request: pytest.FixtureRequest) -> Iterator[MJWSwarmBotsVectorEnv]:
    kwargs = _scenario_kwargs()
    kwargs["reset_settle_time"] = request.param
    env = make_env(TASK,num_envs=2,device="cpu",compile_tensor_operations=False,scenario_kwargs=kwargs)
    try:
        env.reset(seed=7)
        yield env
    finally:
        env.close()


def _place_and_score(env: MJWSwarmBotsVectorEnv, xy: torch.Tensor) -> dict[str, torch.Tensor]:
    runtime = env._scenario_runtime
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,:2]] = xy
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,2]] = runtime.scenario.payload_radius
    return runtime.compute_step_rewards(stable_mask=torch.ones(env.num_envs,dtype=torch.bool)).info


@pytest.mark.parametrize("delta,velocity", [(False,False),(False,True),(True,False),(True,True)])
@pytest.mark.parametrize("inside", [False,True])
@pytest.mark.parametrize("num_goals", [1,3,4])
def test_velocity_layout_normalization_and_nop(delta: bool, velocity: bool, inside: bool, num_goals: int) -> None:
    scenario = make_scenario(
        TASK,include_goal_delta_obs=delta,include_payload_velocity_obs=velocity,include_goal_inside_obs=inside,
        goal_distance_ranges=((1.,2.),)*num_goals,goal_turn_arc_degrees=(90.,)*num_goals,
    )
    base = 13+3*num_goals
    continuous_end = base+2*(delta+velocity)
    dimension = continuous_end+inside
    spaces = scenario.get_single_observation_space()
    assert spaces["global_obs"].shape == (dimension,)
    indices = build_obs_indices(
        env_settings={"scenario":scenario.get_settings()},local_obs_dim=spaces["local_obs"].shape[-1],
        global_obs_dim=dimension,hidden_local_vars_dim=spaces["hidden_local_vars"].shape[-1],hidden_global_vars_dim=1,
    )
    for feature in range(base,continuous_end):
        assert feature in indices.global_scalar_indices
    if inside:
        assert dimension-1 not in indices.global_scalar_indices
    targets = make_next_obs_pred_config(indices)
    assert targets.global_scalar_active_mask_indices == [0]*len(indices.global_scalar_indices)


def test_velocity_observation_updates_and_partial_reset_preserves_other_lane(motion_env: MJWSwarmBotsVectorEnv) -> None:
    runtime = motion_env._scenario_runtime
    velocities = torch.tensor([[.3,-.7],[-1.2,2.3]])
    runtime.bindings.qvel[:,runtime._payload_xy_qvel_indices] = velocities
    runtime._update_payload_obs()
    torch.testing.assert_close(runtime.global_obs[:,24:26],velocities)
    torch.testing.assert_close(runtime.global_obs[:,22:24],runtime.goal_position[:,0]-runtime.payload_position[:,0,:2])
    other = runtime.global_obs[1].clone()
    motion_env.reset(seed=9,options={"reset_mask":torch.tensor([True,False])})
    torch.testing.assert_close(runtime.global_obs[:,24:26],runtime.bindings.qvel[:,runtime._payload_xy_qvel_indices])
    torch.testing.assert_close(runtime.global_obs[1],other,rtol=0.,atol=0.)


@pytest.mark.parametrize("horizontal,vertical,expected", [(.149,0.,1.),(.151,0.,0.),(0.,.149,1.),(0.,.151,0.)])
def test_goal_inside_flag_uses_the_hold_timers_3d_tolerance(
    motion_env: MJWSwarmBotsVectorEnv, horizontal: float, vertical: float, expected: float,
) -> None:
    runtime = motion_env._scenario_runtime
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,:2]] = runtime.goal_sequence[:,0]+torch.tensor([horizontal,0.])
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,2]] = runtime.scenario.payload_radius+vertical
    runtime.compute_step_rewards(stable_mask=torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(runtime.global_obs[:,26],torch.full((2,),expected))
    assert (runtime.goal_inside_samples==int(expected)).all()


def test_goal_inside_flag_tracks_target_switch_and_final_goal(motion_env: MJWSwarmBotsVectorEnv) -> None:
    env = motion_env
    runtime = env._scenario_runtime
    for index in range(3):
        for _ in range(runtime.scenario.goal_hold_steps):
            _place_and_score(env,runtime.goal_sequence[:,index])
            assert (runtime.global_obs[:,26]==1.).all()
        _place_and_score(env,runtime.goal_sequence[:,index])
        assert (runtime.global_obs[:,26]==float(index==2)).all()


def test_broken_hold_refunds_partial_credit_and_repeated_entry_does_not_pay(motion_env: MJWSwarmBotsVectorEnv) -> None:
    env = motion_env
    goal = env._scenario_runtime.goal_sequence[:,0]
    first = _place_and_score(env,goal)
    torch.testing.assert_close(first["hold_progress_reward"],torch.zeros(2))
    second = _place_and_score(env,goal)
    torch.testing.assert_close(second["hold_progress_reward"],torch.full((2,),.5))
    interrupted = _place_and_score(env,goal+torch.tensor([.2,0.]))
    torch.testing.assert_close(interrupted["hold_progress_reward"],torch.full((2,),-.5))
    reentry = _place_and_score(env,goal)
    torch.testing.assert_close(reentry["hold_progress_reward"],torch.zeros(2))
    assert not reentry["goal_1_reached"].any()


@pytest.mark.parametrize("fraction", [.25,.5,1.])
@pytest.mark.parametrize("hold_time", [0.,.06])
def test_full_sequence_retains_its_total_bonus_budget(fraction: float, hold_time: float) -> None:
    kwargs = _scenario_kwargs()
    kwargs.update(goal_hold_reward_fraction=fraction,goal_hold_time=hold_time)
    env = make_env(TASK,num_envs=2,device="cpu",compile_tensor_operations=False,scenario_kwargs=kwargs)
    try:
        env.reset(seed=7)
        runtime = env._scenario_runtime
        total = torch.zeros(2)
        for index,bonus in enumerate((2.,2.,5.)):
            stage_total = torch.zeros(2)
            for _ in range(runtime.scenario.goal_hold_steps+1):
                info = _place_and_score(env,runtime.goal_sequence[:,index])
                stage_total += info["hold_progress_reward"]+info["waypoint_reward"]+info["goal_success_reward"]
            torch.testing.assert_close(stage_total,torch.full((2,),bonus))
            assert (runtime.completed_goals==index+1).all()
            total += stage_total
            if index<2:
                stationary = _place_and_score(env,runtime.goal_sequence[:,index])
                torch.testing.assert_close(stationary["hold_progress_reward"],torch.zeros(2))
        torch.testing.assert_close(total,torch.full((2,),9.))
        repeated = _place_and_score(env,runtime.goal_sequence[:,-1])
        torch.testing.assert_close(repeated["hold_progress_reward"],torch.zeros(2))
    finally:
        env.close()


def test_unstable_lane_neither_earns_nor_loses_hold_credit(motion_env: MJWSwarmBotsVectorEnv) -> None:
    env = motion_env
    runtime = env._scenario_runtime
    _place_and_score(env,runtime.goal_sequence[:,0])
    _place_and_score(env,runtime.goal_sequence[:,0])
    previous = runtime.goal_inside_samples.clone()
    result = runtime.compute_step_rewards(stable_mask=torch.tensor([False,True]))
    assert runtime.goal_inside_samples[0]==previous[0]
    assert result.info["hold_progress_reward"][0]==0.
    assert result.info["goal_1_reached"][1]


def test_terminal_velocity_is_preserved_in_final_observation_before_reset(motion_env: MJWSwarmBotsVectorEnv) -> None:
    env = motion_env
    runtime = env._scenario_runtime
    runtime.completed_goals[:] = 2
    runtime.goal_position[:] = runtime.goal_sequence[:,2:3]
    runtime.goal_inside_samples[:] = runtime.scenario.goal_hold_steps
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,:2]] = runtime.goal_sequence[:,2]
    runtime.bindings.qpos[:,runtime._payload_position_qpos_indices[0,2]] = runtime.scenario.payload_radius
    runtime.bindings.qvel[:,runtime._payload_xy_qvel_indices] = torch.tensor([2.,0.])
    actions = {"actuators":torch.zeros(env.action_space["actuators"].shape),
               "connectors":torch.zeros(env.action_space["connectors"].shape)}
    observations,_,terminated,_,info = env.step(actions)
    assert terminated.all()
    assert (info["final_obs"]["global_obs"][:,24].abs()>.1).all()
    assert (info["final_obs"]["global_obs"][:,26]==1.).all()
    assert (observations["global_obs"][:,26]==0.).all()
    torch.testing.assert_close(observations["global_obs"][:,24:26],runtime.bindings.qvel[:,runtime._payload_xy_qvel_indices])
    assert (observations["global_obs"][:,18]==1.).all()


@pytest.mark.parametrize("fraction", [-.1,1.1,float("nan"),float("inf")])
def test_invalid_hold_reward_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError,match="goal_hold_reward_fraction"):
        make_scenario(TASK,goal_hold_reward_fraction=fraction)


@pytest.mark.cuda
def test_hold_reward_kernel_compiles_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    arguments = dict(
        inside_goal=torch.tensor([True,False],device="cuda"),stable_mask=torch.ones(2,dtype=torch.bool,device="cuda"),
        completed=torch.zeros(2,dtype=torch.long,device="cuda"),inside_samples=torch.tensor([1,2],device="cuda"),
        goals=torch.tensor([[[0.,1.],[0.,2.],[1.,2.]]],device="cuda").expand(2,-1,-1),
        payload_position=torch.tensor([[0.,1.,.3]],device="cuda").expand(2,-1),
        unit_xy=torch.zeros((2,2,2),device="cuda"),units_active_mask=torch.ones((2,2),dtype=torch.bool,device="cuda"),
        payload_progress=torch.zeros((2,1),device="cuda"),approach_progress=torch.zeros(2,device="cuda"),
        required_hold_steps=2,payload_radius=.3,goal_radius=.15,approach_radius=1.25,
        waypoint_reward=2.,success_reward=5.,progress_reward_weight=1.,
        approach_potential="pseudo_huber",approach_softness=.5,hold_reward_fraction=.5,
    )
    expected = _advance_goal_sequence(**arguments)
    compiled = torch.compile(_advance_goal_sequence,fullgraph=True)
    try:
        actual = compiled(**arguments)
        for first,second in zip(actual,expected,strict=True):
            torch.testing.assert_close(first,second)
        torch.testing.assert_close(actual[-1],torch.tensor([.5,-.5],device="cuda"))
    finally:
        torch.compiler.reset()

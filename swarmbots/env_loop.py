import mujoco
from mujoco import MjsBody, MjSpec

from swarmbots.scenarios.base_scenario import BaseScenario
from swarmbots.swarm.base_swarm import BaseSwarm


def do_loop(
        swarm: BaseSwarm,
        scenario: BaseScenario,
        duration: float = 5.0,
):
    spec = MjSpec()
    worldbody: MjsBody = spec.worldbody

    (worldbody
     .add_frame(pos=[scenario.get_start_location()])
     .attach_body(swarm.build_swarm_spec()))

    (worldbody
     .add_frame()
     .attach_body(scenario.build_scenario_spec()))

    model = spec.compile()
    data = mujoco.MjData(model)

    reset = True
    scenario_state = None
    while data.time < duration:
        if reset:
            mujoco.mj_resetData(model, data)
            scenario_state = scenario.reset_scenario(model, data)
            swarm.reset_swarm(model, data)
            reset = False

        mujoco.mj_step(model, data)

        scenario_state, reward, done = scenario.evaluate_state(model, data, scenario_state)

        yield model, data, reward, done, scenario_state

        if done:
            reset = True




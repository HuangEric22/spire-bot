from spire_bot.envs.actions import decode_action, valid_action_mask
from spire_bot.envs.observations import print_observation_debug
from spire_bot.envs.render import print_state as print_state_summary
from spire_bot.envs.simulator_client import SimulatorClient


def print_state(state: dict) -> None:
    """Print the combat state summary plus the full observation feature dump."""
    print_state_summary(state)
    print_observation_debug(state)


with SimulatorClient(no_build=True) as sim:
    state = sim.start_run(character="Silent", seed="smoke-test")
    state = sim.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")

    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))
    
    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(0, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))
    
    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(0, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))
    
    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(1, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

    sim_action = decode_action(0, state)
    print("Action:", sim_action)
    state = sim.act(sim_action.name, **sim_action.args)
    print_state(state)
    print("Valid actions:", valid_action_mask(state))

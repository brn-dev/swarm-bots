from gymnasium import spaces


def get_agent_action_dim(space: spaces.Space) -> int:
    """
    Return the per-agent action dimension for an action space.

    Assumption: Each individual action component space is shaped like (n_agents, n_actions_per_agent)
    """
    if isinstance(space, (spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete)):
        shape = getattr(space, "shape", None)
        if shape is None:
            raise ValueError(f"Expected a shaped space (n_agents, n_actions_per_agent), got {space}")
        if len(shape) != 2:
            raise ValueError(f"Expected shape (n_agents, n_actions_per_agent), got shape={shape} for {space}")
        return int(shape[1])
    raise NotImplementedError(f"{space} space is not supported (expected Box/MultiBinary/MultiDiscrete)")

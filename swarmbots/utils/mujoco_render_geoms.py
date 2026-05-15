import mujoco
import numpy as np


def add_line_geom(
    scene: mujoco.MjvScene,
    *,
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    width: float,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(from_pos, dtype=np.float64),
        np.asarray(to_pos, dtype=np.float64),
    )
    scene.ngeom += 1


def add_payload_centering_boundary_geoms(
    scene: mujoco.MjvScene,
    *,
    tolerance: float,
    half_length: float,
    z: float = 0.025,
) -> None:
    tolerance = float(tolerance)
    x_positions = (0.0,) if tolerance == 0.0 else (-tolerance, tolerance)
    for x_pos in x_positions:
        add_line_geom(
            scene,
            from_pos=(float(x_pos), -float(half_length), z),
            to_pos=(float(x_pos), float(half_length), z),
            rgba=(1.0, 0.16, 0.05, 1.0),
            width=6.0,
        )

import jax.numpy as jnp
import numpy as np


def mjx_quat_to_rot6d(quats: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    if axis != -1:
        quats = jnp.moveaxis(quats, axis, -1)

    q = quats / jnp.maximum(jnp.linalg.norm(quats, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = jnp.moveaxis(q, -1, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot6d = jnp.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
        ],
        axis=-1,
    )
    if axis != -1:
        rot6d = jnp.moveaxis(rot6d, -1, axis)
    return rot6d


def mjx_random_quat_shoemake(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    sqrt1 = np.sqrt(1.0 - u1)
    sqrt2 = np.sqrt(u1)
    theta1 = 2.0 * np.pi * u2
    theta2 = 2.0 * np.pi * u3
    return np.array(
        [
            sqrt2 * np.cos(theta2),
            sqrt1 * np.sin(theta1),
            sqrt1 * np.cos(theta1),
            sqrt2 * np.sin(theta2),
        ],
        dtype=float,
    )

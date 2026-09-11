"""Generic JAX math helpers: clipped Gaussian noise and quaternion utilities."""
import jax
import jax.numpy as jnp


def clipped_normal(rng, shape, std, std_clip=3.0):
    """Helper function to sample clipped Gaussian noise."""
    noise = jax.random.normal(rng, shape) * std
    return jnp.clip(noise, -std_clip * std, std_clip * std)


def quat_to_rotation_matrix(q):
    """
    Convert a drone quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])


def quat_multiply(q1, q2):
    """
    Quaternion multiplication. q1, q2: (..., 4) arrays in [w, x, y, z] format.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return jnp.stack([w, x, y, z], axis=-1)


def add_quat_noise(rng, quat, angle_std=0.01, std_clip=3.0):
    """Adds a small random rotation to a quaternion via axis-angle perturbation,
    quaternion is (..., 4) in [w, x, y, z] format (unit norm)."""
    rng, angle_rng, axis_rng = jax.random.split(rng, 3)
    angle = clipped_normal(angle_rng, quat.shape[:-1] + (1,), angle_std, std_clip)

    axis = jax.random.normal(axis_rng, quat.shape)
    axis = axis / (jnp.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9)

    half = angle / 2.0
    sin_half = jnp.sin(half)
    noise_quat = jnp.concatenate([jnp.cos(half), axis * sin_half], axis=-1)

    final_quat = quat_multiply(noise_quat, quat)
    final_quat = final_quat / (jnp.linalg.norm(final_quat, axis=-1, keepdims=True) + 1e-8)

    return final_quat, rng

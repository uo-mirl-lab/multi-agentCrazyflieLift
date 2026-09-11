"""Debug helpers for verifying MJX domain-randomization noise (mass/inertia/
actuator gain jitter applied on env.reset)."""
import numpy as np


def print_array_diff(name, a, b):
    """Print only the entries where arrays `a` and `b` differ."""
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")

    diff_mask = ~np.isclose(a, b, 1e-8)

    if not np.any(diff_mask):
        print(f"\n{name}: no change")
        return

    print(f"\n{name}:")
    for idx in zip(*np.where(diff_mask)):
        old = a[idx]
        new = b[idx]
        print(f"  {old:.6g} --> {new:.6g}")


def model_physics_compare(mjx_model, mjx_model_2, seed):
    """Print mass/inertia/actuator-gain differences between two MJX models for a given seed."""
    print("Domain randomization test for seed", seed)

    print_array_diff("Mass randomization", mjx_model.body_mass, mjx_model_2.body_mass)
    print_array_diff("Inertia randomization", mjx_model.body_inertia, mjx_model_2.body_inertia)
    print_array_diff("Actuator gain randomization", mjx_model.actuator_gainprm, mjx_model_2.actuator_gainprm)

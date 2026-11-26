import torch


@torch.no_grad()
def sample_euler(velocity, x0, steps, *, return_trajectory=False):
    """Move 2D points from t=0 to t=1 using uniform Euler steps.

    Returns endpoints of shape (N, 2). If return_trajectory is true,
    returns all states of shape (steps + 1, N, 2), starting with x0.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if x0.ndim != 2 or x0.shape[1] != 2:
        raise ValueError("x0 must have shape (N, 2).")

    x = x0.clone()
    dt = 1.0 / steps
    trajectory = [x] if return_trajectory else None

    for step in range(steps):
        t = x.new_full((len(x), 1), step * dt)
        x = x + dt * velocity(x, t)
        if trajectory is not None:
            trajectory.append(x)

    return x if trajectory is None else torch.stack(trajectory)

import torch
from torch import nn


class VelocityMLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x, t):
        return self.network(torch.cat((x, t), dim=1))


def velocity_loss(model, x0, x1, t):
    x_t = (1 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted_velocity = model(x_t, t)
    return nn.functional.mse_loss(predicted_velocity, target_velocity)


def fit_velocity(model, x0, x1, *, updates, batch_size=256, lr=1e-3, seed=42):
    """Train on paired CPU tensors of shape (num_points, 2)."""
    if x0.shape != x1.shape or x0.ndim != 2 or x0.shape[1] != 2 or len(x0) == 0:
        raise ValueError("x0 and x1 must have the same nonempty shape (N, 2).")

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    for _ in range(updates):
        indices = torch.randint(len(x0), (batch_size,), generator=generator)
        t = torch.rand((batch_size, 1), generator=generator, dtype=x0.dtype)

        optimizer.zero_grad()
        loss = velocity_loss(model, x0[indices], x1[indices], t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

from .sde import SDE
import torch


class EulerMaruyama:
    def __init__(self, sde: SDE) -> None:
        self.sde = sde

    def step(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Simulate one step of SDE.

        Args:
            x: The location. Shape (b, d).
            t: The time. Shape (b,).
            h: The time step. Shape (b,).

        Returns:
            The location at time t + h. Shape (b, d).
        """
        dWt = torch.randn_like(x) * torch.sqrt(h)[:, None]  # shape (b, d)
        return x + h[:, None] * self.sde.u(x, t) + self.sde.sigma(x, t) * dWt

    def simulate_trajectory(self, x0: torch.Tensor, t: torch.Tensor, whole: bool = False) -> torch.Tensor:
        """
        Simulate trajectory of ODE.

        Args:
            x0: The initial value. Shape (b, d).
            t: The time. t[0] = 0. Shape (b, l).
            whole: Return the whole trajectory.

        Returns:
            The whole trajectory if whole=True, shape (b, l, d), the end point otherwise, shape (b, d).
        """
        trajectory = [x0]
        xi = x0.clone()
        l = t.shape[1]

        for i in range(1, l):
            xi = self.step(xi, t[:, i-1], t[:, i] - t[:, i-1])
            if whole:
                trajectory.append(xi.clone())

        if whole:
            return torch.stack(trajectory, dim=1) 
        
        return xi
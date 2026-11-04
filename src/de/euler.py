from .ode import ODE
import torch


class Euler:
    def __init__(self, ode: ODE) -> None:
        self.ode = ode

    def step(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Simulate one step of ODE.

        Args:
            x: The location. Shape (b, d).
            t: The time. Shape (b,).
            h: The time step. Shape (b,).

        Returns:
            The location at time t + h. Shape (b, d).
        """
        return x + h[:, None] * self.ode.u(x, t)

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
            xi += (t[:, i:i+1] - t[:, i-1:i]) * self.ode.u(xi, t[:, i-1])
            if whole:
                trajectory.append(xi.clone())

        if whole:
            return torch.stack(trajectory, dim=1) 
        
        return xi
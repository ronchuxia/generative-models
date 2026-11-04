from abc import ABC, abstractmethod
import torch


class ODE:
    @abstractmethod
    def u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity ut at state x at time t.

        Args:
            x: The state. Shape (b, d).
            t: The time. Shape (b,).

        Returns:
            The velocity ut at state x at time t. Shape (b, d).
        """
        pass

    def flow(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute state xt given initial state x0.

        Args:
            x0: The initial state. Shape (b, d).
            t: The time. Shape (b, l).

        Returns:
            The state xt. Shape (b, l, d).
        """
        raise NotImplementedError("This ODE doesn't have a closed form flow.")


class LinearVectorFieldODE(ODE):
    def __init__(self, theta: float) -> None:
        super().__init__()
        self.theta = theta
    
    def u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return - self.theta * x

    def flow(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(- self.theta * t)[:, :, None] * x0[:, None, :]
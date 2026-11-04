from abc import ABC, abstractmethod
import torch


class SDE:
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

    @abstractmethod
    def sigma(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient sigmat at state x at time t.

        Args:
            x: The state. Shape (b, d).
            t: The time. Shape (b,).
        
        Returns:
            The diffusion coefficient sigmat at state x at time t. Shape (b, d).
        """
        pass


class WienerProcessSDE(SDE):
    def __init__(self) -> None:
        super().__init__()

    def u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def sigma(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class OUProcessSDE(SDE):
    def __init__(self, theta: float, sigma: float) -> None:
        super().__init__()
        self._theta = theta
        self._sigma = sigma
    
    def u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return - self._theta * x

    def sigma(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self._sigma)
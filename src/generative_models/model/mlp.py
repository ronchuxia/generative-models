import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: list) -> None:
        """
        Initialize an MLP.

        Args:
            in_features: Number of input features of the input layer.
            hidden_features: Number of hidden features of each layer.
        """
        super().__init__()
        
        # Input layer
        layers = [
            nn.Linear(in_features, hidden_features[0]),
            nn.ReLU(),
        ]

        # Hidden layers
        num_hidden_layers = len(hidden_features)
        for i in range(num_hidden_layers-2):
            layers.append(nn.Linear(hidden_features[i], hidden_features[i+1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_features[-2], hidden_features[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class FlowMatchingMLP(nn.Module):
    def __init__(self, x_dim: int, hidden_features: list) -> None:
        """
        Initialize an MLP for flow matching.
        
        Args:
            x_dim: Dimensions of x.
            hidden_features: Number of hidden features of each layer.
        """
        super().__init__()

        self.mlp = MLP(x_dim + 1, hidden_features)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Predict u(x, t).

        Args:
            x: State. Shape (b, d).
            t: Time. Shape (b, 1).

        Returns:
            u(x, t). Shape (b, d).
        """
        return self.mlp(torch.concat([x, t], dim=1))

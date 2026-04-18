import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    """
    Simple Physics-Informed Neural Network (PINN)

    Architecture:
        Input:  dimension controlled by input_dim
                  - input_dim=1 : steady problems  -> input is (x,)
                  - input_dim=2 : time-dependent   -> input is (x, t)
        Hidden: configurable depth/width, Tanh activation
        Output: u(x) or u(x,t)

    Notes:
        - The old t=x dummy-variable hack is removed.
        - Caller is responsible for passing the correct number of tensors
          to forward().
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 50, n_layers: int = 2):
        """
        Parameters
        ----------
        input_dim  : number of input features (1 for steady, 2 for time-dep.)
        hidden_dim : neurons per hidden layer
        n_layers   : number of hidden layers
        """
        super(SimplePINN, self).__init__()

        if input_dim not in (1, 2):
            raise ValueError(f"input_dim must be 1 or 2, got {input_dim}")

        self.input_dim = input_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, *coords):
        """
        Forward pass.

        Parameters
        ----------
        *coords : tensors of shape (N, 1) each.
                  Pass (x,)    for steady  problems (input_dim=1).
                  Pass (x, t)  for time-dep problems (input_dim=2).

        Returns
        -------
        u : predicted solution, shape (N, 1)
        """
        if len(coords) != self.input_dim:
            raise ValueError(
                f"Model expects {self.input_dim} input tensor(s), "
                f"got {len(coords)}."
            )
        inputs = torch.cat(list(coords), dim=1)
        return self.net(inputs)

    def _initialize_weights(self):
        """Xavier uniform init — standard for PINNs."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

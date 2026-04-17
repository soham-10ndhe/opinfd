import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    """
    Simple Physics-Informed Neural Network (PINN)

    Architecture:
        Input:  (x, t)
        Hidden: 2 layers, 50 neurons each, Tanh activation
        Output: u(x)

    Notes:
        - t is a dummy variable for steady problems (t = x)
        - Designed to be easily extendable for time-dependent PDEs
    """

    def __init__(self):
        super(SimplePINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

        self._initialize_weights()

    def forward(self, x, t):
        """
        Forward pass

        Parameters:
            x : spatial input tensor (N,1)
            t : time input tensor (N,1)

        Returns:
            u : predicted solution (N,1)
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def _initialize_weights(self):
        """
        Xavier initialization for better PINN convergence
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
# OPINFD - Open-source Physics Informed Neural Fluid Dynamics

from .models import SimplePINN
from .physics import poisson_loss
from .trainer import train_case

__all__ = [
    "SimplePINN",
    "poisson_loss",
    "train_case",
]

__version__ = "0.1.0"
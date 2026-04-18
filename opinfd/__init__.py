# OPINFD - Open-source Physics Informed Neural Fluid Dynamics

from .models import SimplePINN
from .physics import PoissonPDE, BurgersPDE, get_pde
from .trainer import train_case

__all__ = [
    "SimplePINN",
    "PoissonPDE",
    "BurgersPDE",
    "get_pde",
    "train_case",
]

__version__ = "0.2.0"

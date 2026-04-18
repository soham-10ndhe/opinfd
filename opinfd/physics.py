"""
opinfd/physics.py
-----------------
Plugin-style PDE system.

Each PDE is a class inheriting BasePDE and implements:
    residual(model, x_col, ...)  ->  scalar residual loss
    bc_ic_loss(model, ...)       ->  scalar boundary / initial condition loss
    exact(x, t=None)             ->  numpy array of exact solution (for validation)

Trainer calls only the BasePDE interface, so new PDEs never require
changes to trainer.py.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


# ===========================================================================
# Base class
# ===========================================================================

class BasePDE(ABC):
    """Abstract base for all PDE plugins."""

    # ---- must implement ---------------------------------------------------
    @abstractmethod
    def residual_loss(self, model, *collocation_pts):
        """PDE residual MSE at collocation points."""

    @abstractmethod
    def bc_ic_loss(self, model, *boundary_pts):
        """Boundary / initial condition MSE."""

    @abstractmethod
    def exact(self, *pts_numpy):
        """Exact solution as a numpy array (for validation)."""

    # ---- convenience ------------------------------------------------------
    def total_loss(self, model, collocation_pts, boundary_pts):
        return (self.residual_loss(model, *collocation_pts)
                + self.bc_ic_loss(model, *boundary_pts))


# ===========================================================================
# 1-D Poisson   -u_xx = pi^2 sin(pi x),  x in [-1,1]
# Exact: u(x) = sin(pi x)
# ===========================================================================

class PoissonPDE(BasePDE):
    """
    1-D Poisson equation:
        -u_xx = pi^2 * sin(pi*x),   x in [x_min, x_max]
        u(x_min) = u(x_max) = 0

    Exact solution: u = sin(pi*x)
    """

    def residual_loss(self, model, x_col):
        x_col = x_col.requires_grad_(True)
        u = model(x_col)                          # steady -> input_dim=1

        u_x  = torch.autograd.grad(u,  x_col, torch.ones_like(u),  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, torch.ones_like(u_x), create_graph=True)[0]

        f = u_xx + (np.pi ** 2) * torch.sin(np.pi * x_col)
        return torch.mean(f ** 2)

    def bc_ic_loss(self, model, x_bc):
        u_bc = model(x_bc)
        return torch.mean(u_bc ** 2)

    def exact(self, x_np):
        return np.sin(np.pi * x_np)


# ===========================================================================
# 1-D Burgers   u_t + u*u_x = nu * u_xx,  x in [-1,1], t in [0,1]
# nu = 0.01/pi
# IC : u(x,0) = -sin(pi*x)
# BC : u(-1,t) = u(1,t) = 0
# ===========================================================================

class BurgersPDE(BasePDE):
    """
    1-D viscous Burgers equation:
        u_t + u * u_x  =  nu * u_xx
        IC: u(x, 0) = -sin(pi*x)
        BC: u(-1, t) = u(1, t) = 0

    nu = 0.01 / pi  (classic PINN benchmark, Raissi et al. 2019)

    Note: No closed-form exact solution is used here.
          Validation uses a high-resolution finite-difference reference
          baked in below.
    """

    def __init__(self, nu: float = 0.01 / np.pi):
        self.nu = nu

    # ------------------------------------------------------------------
    def residual_loss(self, model, x_col, t_col):
        x_col = x_col.requires_grad_(True)
        t_col = t_col.requires_grad_(True)

        u = model(x_col, t_col)                   # time-dep -> input_dim=2

        u_t  = torch.autograd.grad(u,  t_col, torch.ones_like(u),  create_graph=True)[0]
        u_x  = torch.autograd.grad(u,  x_col, torch.ones_like(u),  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_col, torch.ones_like(u_x), create_graph=True)[0]

        f = u_t + u * u_x - self.nu * u_xx
        return torch.mean(f ** 2)

    # ------------------------------------------------------------------
    def bc_ic_loss(self, model, x_ic, t_ic, x_bc, t_bc):
        """
        IC loss: u(x, 0) = -sin(pi*x)
        BC loss: u(-1, t) = 0,  u(1, t) = 0
        """
        u_ic  = model(x_ic, t_ic)
        u_ic_exact = -torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic - u_ic_exact) ** 2)

        u_bc  = model(x_bc, t_bc)
        loss_bc = torch.mean(u_bc ** 2)

        return loss_ic + loss_bc

    # ------------------------------------------------------------------
    def exact(self, x_np, t_np):
        """
        Pseudo-exact via scipy solve_ivp finite-difference (Method of Lines).
        Returns u(x, t) as a 2-D array [Nx, Nt] — or 1-D if x_np/t_np are
        paired vectors.
        """
        # Paired (x_i, t_i) evaluation — used for scatter-plot validation
        from scipy.interpolate import RegularGridInterpolator

        ref = self._fd_reference()          # dict with x, t, U grids
        interp = RegularGridInterpolator(
            (ref["x"], ref["t"]), ref["U"],
            method="linear", bounds_error=False, fill_value=None
        )
        pts = np.stack([x_np.ravel(), t_np.ravel()], axis=1)
        return interp(pts).reshape(x_np.shape)

    def _fd_reference(self, Nx=256, Nt=1000):
        """
        Method-of-Lines finite difference reference solution.
        Cached on first call.
        """
        if hasattr(self, "_fd_cache"):
            return self._fd_cache

        nu  = self.nu
        x   = np.linspace(-1, 1, Nx)
        t   = np.linspace(0,  1, Nt)
        dx  = x[1] - x[0]
        dt  = t[1] - t[0]

        U = np.zeros((Nx, Nt))
        U[:, 0] = -np.sin(np.pi * x)          # IC

        for n in range(Nt - 1):
            u = U[:, n].copy()
            # central difference in space
            u_x  = np.zeros_like(u)
            u_xx = np.zeros_like(u)
            u_x [1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
            rhs = -u * u_x + nu * u_xx
            U[:, n+1]    = u + dt * rhs
            U[0,  n+1]   = 0.0                 # BC left
            U[-1, n+1]   = 0.0                 # BC right

        self._fd_cache = {"x": x, "t": t, "U": U}
        return self._fd_cache


# ===========================================================================
# Registry — maps YAML pde_type string -> class
# ===========================================================================

PDE_REGISTRY = {
    "poisson_1d": PoissonPDE,
    "burgers_1d": BurgersPDE,
}


def get_pde(config: dict) -> BasePDE:
    """
    Instantiate the correct PDE from a case config dict.

    The config must contain a ``pde_type`` key matching a key in PDE_REGISTRY.
    Optional PDE-specific kwargs (e.g. ``nu``) are forwarded if present.
    """
    pde_type = config.get("pde_type")
    if pde_type is None:
        raise KeyError("case.yaml must contain a 'pde_type' field.")
    if pde_type not in PDE_REGISTRY:
        raise ValueError(
            f"Unknown pde_type '{pde_type}'. "
            f"Available: {list(PDE_REGISTRY.keys())}"
        )

    cls = PDE_REGISTRY[pde_type]

    # Forward any PDE-specific kwargs present in config
    import inspect
    valid_keys = inspect.signature(cls.__init__).parameters.keys() - {"self"}
    kwargs = {k: config[k] for k in valid_keys if k in config}

    return cls(**kwargs)

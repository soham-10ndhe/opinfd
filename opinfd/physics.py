import torch
import numpy as np

def poisson_loss(model, x_col, x_bc):
    x_col.requires_grad_(True)
    t_col = x_col

    u = model(x_col, t_col)

    u_x = torch.autograd.grad(u, x_col, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_col, torch.ones_like(u_x), create_graph=True)[0]

    f = u_xx + (np.pi**2) * torch.sin(np.pi * x_col)
    loss_pde = torch.mean(f**2)

    t_bc = x_bc
    u_bc = model(x_bc, t_bc)
    loss_bc = torch.mean(u_bc**2)

    return loss_pde + loss_bc
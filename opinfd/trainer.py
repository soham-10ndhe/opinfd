"""
opinfd/trainer.py
-----------------
Generic trainer that works with any BasePDE plugin.

Key upgrades over Stage 1:
  - PDE-agnostic: all physics delegated to the PDE object
  - Adaptive RAR: multiple rounds, each adding rar_points high-residual pts
  - Loss/error vs epoch plot saved alongside solution plot
  - Proper (x, t) collocation for time-dependent problems
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from opinfd.models import SimplePINN
from opinfd.physics import get_pde, PoissonPDE, BurgersPDE
from opinfd.utils import create_results_dir, save_metrics


# ===========================================================================
# Helpers
# ===========================================================================

def _make_collocation(config, device):
    """
    Build initial collocation + boundary/IC tensors for whatever PDE type.

    Returns
    -------
    col_pts  : list of tensors passed to pde.residual_loss(*col_pts)
    bc_pts   : list of tensors passed to pde.bc_ic_loss(*bc_pts)
    pool_fn  : callable(n) -> list[tensor]  — samples a random pool for RAR
    """
    x_min = config["domain"]["x_min"]
    x_max = config["domain"]["x_max"]
    N_col = config["N_col"]
    N_bc  = config["N_bc"]

    pde_type = config["pde_type"]

    if pde_type == "poisson_1d":
        x_col = torch.FloatTensor(N_col, 1).uniform_(x_min, x_max).to(device)

        x_bc = torch.cat([
            torch.full((N_bc // 2, 1), x_min),
            torch.full((N_bc // 2, 1), x_max),
        ], dim=0).to(device)

        col_pts = [x_col]
        bc_pts  = [x_bc]

        def pool_fn(n):
            xp = torch.FloatTensor(n, 1).uniform_(x_min, x_max).to(device)
            return [xp]

    elif pde_type == "burgers_1d":
        t_min = config["domain"].get("t_min", 0.0)
        t_max = config["domain"].get("t_max", 1.0)

        x_col = torch.FloatTensor(N_col, 1).uniform_(x_min, x_max).to(device)
        t_col = torch.FloatTensor(N_col, 1).uniform_(t_min, t_max).to(device)

        # IC: t=0, x random
        x_ic = torch.FloatTensor(N_bc, 1).uniform_(x_min, x_max).to(device)
        t_ic = torch.zeros(N_bc, 1).to(device)

        # BC: x = ±1, t random
        x_bc = torch.cat([
            torch.full((N_bc // 2, 1), x_min),
            torch.full((N_bc // 2, 1), x_max),
        ], dim=0).to(device)
        t_bc = torch.FloatTensor(N_bc, 1).uniform_(t_min, t_max).to(device)

        col_pts = [x_col, t_col]
        bc_pts  = [x_ic, t_ic, x_bc, t_bc]

        def pool_fn(n):
            xp = torch.FloatTensor(n, 1).uniform_(x_min, x_max).to(device)
            tp = torch.FloatTensor(n, 1).uniform_(t_min, t_max).to(device)
            return [xp, tp]

    else:
        raise ValueError(f"Unsupported pde_type: {pde_type}")

    return col_pts, bc_pts, pool_fn


def _residual_sampling(pde, model, pool_pts, top_k):
    """
    RAR: score a pool of candidate points and return the top_k highest residual.
    Works for any PDE — pool_pts must have requires_grad set inside residual.
    """
    # Temporarily enable grad for pool points
    enabled = [p.requires_grad_(True) for p in pool_pts]

    with torch.enable_grad():
        r_loss = pde.residual_loss(model, *enabled)

    # We need pointwise residuals, not the mean — re-run without mean
    # Recompute per-point residual magnitudes
    for p in enabled:
        p.requires_grad_(True)

    # Per-point residual via grad of sum w.r.t. each input
    # Simpler: recompute residual explicitly per PDE type
    with torch.enable_grad():
        pts = [p.requires_grad_(True) for p in pool_pts]
        pointwise = _pointwise_residual(pde, model, pts)

    idx = torch.topk(pointwise.squeeze(), min(top_k, pointwise.shape[0]))[1]

    selected = [p[idx].detach() for p in pool_pts]
    return selected


def _pointwise_residual(pde, model, pts):
    """Return |residual|^1 per point (not meaned)."""
    if isinstance(pde, PoissonPDE):
        x = pts[0].requires_grad_(True)
        u = model(x)
        u_x  = torch.autograd.grad(u,  x, torch.ones_like(u),  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return torch.abs(u_xx + (np.pi**2) * torch.sin(np.pi * x)).detach()

    elif isinstance(pde, BurgersPDE):
        x, t = pts[0].requires_grad_(True), pts[1].requires_grad_(True)
        u    = model(x, t)
        u_t  = torch.autograd.grad(u,  t, torch.ones_like(u),  create_graph=True)[0]
        u_x  = torch.autograd.grad(u,  x, torch.ones_like(u),  create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return torch.abs(u_t + u * u_x - pde.nu * u_xx).detach()

    else:
        raise NotImplementedError(f"Pointwise residual not implemented for {type(pde)}")


# ===========================================================================
# Main trainer
# ===========================================================================

def train_case(config, case_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OPINFD] Device: {device}")

    pde = get_pde(config)
    print(f"[OPINFD] PDE: {config['pde_type']}")

    # Determine model input_dim from pde_type
    input_dim = 1 if config["pde_type"] == "poisson_1d" else 2
    model = SimplePINN(input_dim=input_dim).to(device)

    col_pts, bc_pts, pool_fn = _make_collocation(config, device)
    results_dir = create_results_dir(case_dir)

    loss_history  = []   # (epoch, loss)
    phase_markers = {}   # label -> epoch index for vertical lines on plot

    # =======================================================================
    # Phase 1: Adam
    # =======================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs_adam = config["epochs_adam"]
    print(f"[OPINFD] Phase 1: Adam ({epochs_adam} epochs)")

    phase_markers["Adam start"] = 0

    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss = pde.total_loss(model, col_pts, bc_pts)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"  [Adam {epoch:5d}] Loss: {loss.item():.6e}  "
                  f"  col_pts: {col_pts[0].shape[0]}")

    # =======================================================================
    # Adaptive RAR (multiple rounds)
    # =======================================================================
    rar_rounds  = config.get("rar_rounds", 3)
    rar_points  = config.get("rar_points", 2000)
    pool_size   = config.get("rar_pool_size", 20000)

    print(f"[OPINFD] Adaptive RAR: {rar_rounds} rounds x {rar_points} pts/round")

    for rnd in range(rar_rounds):
        phase_markers[f"RAR {rnd+1}"] = len(loss_history)

        pool_pts = pool_fn(pool_size)
        new_pts  = _residual_sampling(pde, model, pool_pts, top_k=rar_points)

        # Append new points to collocation set
        col_pts = [torch.cat([col_pts[i], new_pts[i]], dim=0)
                   for i in range(len(col_pts))]

        print(f"  [RAR round {rnd+1}] col_pts now: {col_pts[0].shape[0]}")

        # Brief Adam refinement after each RAR round
        rar_refine_epochs = config.get("rar_refine_epochs", 500)
        for epoch in range(rar_refine_epochs):
            optimizer.zero_grad()
            loss = pde.total_loss(model, col_pts, bc_pts)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

    # =======================================================================
    # Phase 2: L-BFGS
    # =======================================================================
    phase_markers["L-BFGS"] = len(loss_history)
    print(f"[OPINFD] Phase 2: L-BFGS ({config['epochs_lbfgs']} iters)")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config["epochs_lbfgs"],
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=50,
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = pde.total_loss(model, col_pts, bc_pts)
        loss.backward()
        loss_history.append(loss.item())
        return loss

    optimizer_lbfgs.step(closure)

    # =======================================================================
    # Validation & plots
    # =======================================================================
    rel_l2 = _validate_and_plot(pde, model, config, results_dir, device)

    _plot_loss_history(loss_history, phase_markers, results_dir)

    # Save model + metrics
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    save_metrics(os.path.join(results_dir, "metrics.json"), {
        "rel_l2_error": float(rel_l2),
        "final_col_pts": int(col_pts[0].shape[0]),
        "rar_rounds": rar_rounds,
    })

    print(f"[OPINFD] Rel L2 Error: {rel_l2:.6e}")
    return rel_l2


# ===========================================================================
# Validation
# ===========================================================================

def _validate_and_plot(pde, model, config, results_dir, device):
    x_min = config["domain"]["x_min"]
    x_max = config["domain"]["x_max"]

    if isinstance(pde, PoissonPDE):
        x_test = torch.linspace(x_min, x_max, 1000).view(-1, 1).to(device)
        with torch.no_grad():
            u_pred = model(x_test).cpu().numpy()

        x_np    = x_test.cpu().numpy()
        u_exact = pde.exact(x_np)
        rel_l2  = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

        plt.figure(figsize=(7, 4))
        plt.plot(x_np, u_exact, label="Exact", lw=2)
        plt.plot(x_np, u_pred, "--", label="PINN", lw=2)
        plt.xlabel("x"); plt.ylabel("u")
        plt.title(f"1-D Poisson  |  Rel L2 = {rel_l2:.3e}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot.png"), dpi=150)
        plt.close()

    elif isinstance(pde, BurgersPDE):
        t_min = config["domain"].get("t_min", 0.0)
        t_max = config["domain"].get("t_max", 1.0)

        Nx, Nt = 200, 100
        x_v = np.linspace(x_min, x_max, Nx)
        t_v = np.linspace(t_min, t_max, Nt)
        XX, TT = np.meshgrid(x_v, t_v, indexing="ij")   # (Nx, Nt)

        x_flat = torch.FloatTensor(XX.ravel()).view(-1, 1).to(device)
        t_flat = torch.FloatTensor(TT.ravel()).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred_flat = model(x_flat, t_flat).cpu().numpy()

        U_pred = u_pred_flat.reshape(Nx, Nt)
        U_exact = pde.exact(XX, TT)

        rel_l2 = (np.linalg.norm(U_pred - U_exact)
                  / np.linalg.norm(U_exact))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        kw = dict(aspect="auto", origin="lower",
                  extent=[t_min, t_max, x_min, x_max], cmap="RdBu_r")

        im0 = axes[0].imshow(U_exact, **kw)
        axes[0].set_title("Exact (FD ref)"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(U_pred, **kw)
        axes[1].set_title("PINN prediction"); axes[1].set_xlabel("t")
        plt.colorbar(im1, ax=axes[1])

        err = np.abs(U_pred - U_exact)
        im2 = axes[2].imshow(err, aspect="auto", origin="lower",
                             extent=[t_min, t_max, x_min, x_max], cmap="hot")
        axes[2].set_title(f"|Error|  Rel L2={rel_l2:.3e}"); axes[2].set_xlabel("t")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot.png"), dpi=150)
        plt.close()

    else:
        raise NotImplementedError(f"Validation not implemented for {type(pde)}")

    return rel_l2


# ===========================================================================
# Loss + phase-marker plot
# ===========================================================================

def _plot_loss_history(loss_history, phase_markers, results_dir):
    epochs = np.arange(len(loss_history))
    losses = np.array(loss_history)

    plt.figure(figsize=(9, 4))
    plt.semilogy(epochs, losses, lw=1.2, color="steelblue", label="Total loss")

    colors = ["#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#2980b9"]
    for i, (label, ep) in enumerate(phase_markers.items()):
        plt.axvline(ep, color=colors[i % len(colors)],
                    linestyle="--", lw=1.2, label=label)

    plt.xlabel("Iteration"); plt.ylabel("Loss (log)")
    plt.title("Training loss history")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_history.png"), dpi=150)
    plt.close()
    print("[OPINFD] Saved loss_history.png")

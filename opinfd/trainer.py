import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from opinfd.models import SimplePINN
from opinfd.physics import poisson_loss
from opinfd.utils import create_results_dir, save_metrics


# =========================
# RAR: Add high-residual points
# =========================
def residual_sampling(model, x_pool, device, top_k=2000):
    x_pool.requires_grad_(True)
    t_pool = x_pool

    u = model(x_pool, t_pool)

    u_x = torch.autograd.grad(u, x_pool, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pool, torch.ones_like(u_x), create_graph=True)[0]

    res = torch.abs(u_xx + (np.pi**2) * torch.sin(np.pi * x_pool))
    idx = torch.topk(res.squeeze(), top_k)[1]

    return x_pool[idx].detach()


# =========================
# Training
# =========================
def train_case(config, case_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OPINFD] Device: {device}")

    x_min = config["domain"]["x_min"]
    x_max = config["domain"]["x_max"]

    model = SimplePINN().to(device)

    # ------------------------
    # Initial Sampling
    # ------------------------
    x_col = torch.FloatTensor(config["N_col"], 1).uniform_(x_min, x_max).to(device)

    x_bc_left = torch.full((config["N_bc"]//2, 1), x_min)
    x_bc_right = torch.full((config["N_bc"]//2, 1), x_max)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0).to(device)

    results_dir = create_results_dir(case_dir)

    # =========================
    # Phase 1: Adam
    # =========================
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print("[OPINFD] Phase 1: Adam")

    for epoch in range(config["epochs_adam"]):
        optimizer.zero_grad()
        loss = poisson_loss(model, x_col, x_bc)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"[Adam {epoch}] Loss: {loss.item():.6e}")

    # =========================
    # RAR Step
    # =========================
    print("[OPINFD] Applying RAR...")

    x_pool = torch.FloatTensor(20000, 1).uniform_(x_min, x_max).to(device)
    x_new = residual_sampling(model, x_pool, device, top_k=config["rar_points"])

    x_col = torch.cat([x_col, x_new], dim=0)

    print(f"[OPINFD] Added {len(x_new)} new points")

    # =========================
    # Phase 2: LBFGS
    # =========================
    print("[OPINFD] Phase 2: LBFGS")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config["epochs_lbfgs"],
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=50
    )

    def closure():
        optimizer.zero_grad()
        loss = poisson_loss(model, x_col, x_bc)
        loss.backward()
        return loss

    optimizer.step(closure)

    # =========================
    # Validation
    # =========================
    x_test = torch.linspace(x_min, x_max, 1000).view(-1, 1).to(device)
    t_test = x_test

    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy()

    x_np = x_test.cpu().numpy()
    u_exact = np.sin(np.pi * x_np)

    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

    print(f"[OPINFD] Rel L2 Error: {rel_l2:.6e}")

    # Plot
    plt.figure()
    plt.plot(x_np, u_exact, label="Exact")
    plt.plot(x_np, u_pred, "--", label="Predicted")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "plot.png"), dpi=300)
    plt.close()

    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))

    save_metrics(os.path.join(results_dir, "metrics.json"), {
        "rel_l2_error": float(rel_l2)
    })

    return rel_l2
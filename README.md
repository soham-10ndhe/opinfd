# OPINFD — Open-source Physics-Informed Neural Fluid Dynamics

> OpenFOAM-style YAML-driven PINN framework · PyTorch · Adaptive RAR · Plugin PDEs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## What is OPINFD?

OPINFD is a lightweight, extensible PINN (Physics-Informed Neural Network) framework inspired by OpenFOAM's case-directory workflow. Each simulation is a self-contained folder with a single `case.yaml` — no code changes needed to run a new PDE.

**Stage 2 features:**
- ✅ Plugin PDE system — add new equations without touching the trainer
- ✅ Fixed `t = x` dummy-variable hack — honest `input_dim` per problem type
- ✅ Adaptive RAR — multiple rounds of residual-based point enrichment
- ✅ Loss vs. epoch plot with phase markers (Adam / RAR rounds / L-BFGS)
- ✅ Two benchmark cases: 1-D Poisson and 1-D Burgers

---

## Project Structure

```
OPINFD/
│
├── opinfd/                   # Core library
│   ├── __init__.py           #   Public API + version
│   ├── models.py             #   SimplePINN (input_dim-aware)
│   ├── physics.py            #   BasePDE, PoissonPDE, BurgersPDE, registry
│   ├── trainer.py            #   Generic adaptive trainer
│   └── utils.py              #   I/O helpers
│
├── cases/                    # One folder per simulation case
│   ├── poisson_1d/
│   │   └── case.yaml         #   1-D Poisson config
│   └── burgers_1d/
│       └── case.yaml         #   1-D Burgers config (NEW)
│
├── scripts/
│   └── run_case.py           # CLI entry point
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup_opinfd.py           # One-command conda setup + run
```

After a run, each case folder gains a `results/` directory:

```
cases/poisson_1d/results/
├── plot.png            # Solution comparison (exact vs PINN)
├── loss_history.png    # Loss curve with phase markers
├── model.pth           # Saved model weights  [git-ignored]
└── metrics.json        # Rel L2 error + final collocation count
```

---

## Installation

### Option A — One command (conda)

```bash
git clone https://github.com/<your-username>/OPINFD.git
cd OPINFD
python setup_opinfd.py                        # runs poisson_1d by default
python setup_opinfd.py cases/burgers_1d       # or specify a case
```

### Option B — Manual (pip / venv)

```bash
git clone https://github.com/<your-username>/OPINFD.git
cd OPINFD

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running a Case

```bash
# From the repo root
python -m scripts.run_case cases/poisson_1d
python -m scripts.run_case cases/burgers_1d
```

Console output example:
```
[OPINFD] Device: cuda
[OPINFD] PDE: burgers_1d
[OPINFD] Phase 1: Adam (8000 epochs)
  [Adam     0] Loss: 4.231456e-01   col_pts: 10000
  [Adam   500] Loss: 8.714523e-03   col_pts: 10000
  ...
[OPINFD] Adaptive RAR: 3 rounds x 2000 pts/round
  [RAR round 1] col_pts now: 12000
  [RAR round 2] col_pts now: 14000
  [RAR round 3] col_pts now: 16000
[OPINFD] Phase 2: L-BFGS (1000 iters)
[OPINFD] Rel L2 Error: 3.214e-03
[SUCCESS] Target achieved  (err=3.214e-03 < 1.000e-02)
```

---

## Case Configuration (`case.yaml`)

All parameters live in `case.yaml`. No Python edits needed for new runs.

### Poisson 1-D (`cases/poisson_1d/case.yaml`)

```yaml
case_name: "1D_Poisson_RAR"
pde_type:  "poisson_1d"

domain:
  x_min: -1.0
  x_max:  1.0

N_col: 8000          # initial collocation points
N_bc:  100           # boundary condition points

epochs_adam:  5000
epochs_lbfgs: 500
lr: 0.001

rar_rounds:        3      # number of RAR enrichment rounds
rar_points:     2000      # new pts added per round
rar_pool_size: 20000      # candidate pool size for RAR scoring
rar_refine_epochs: 500    # Adam epochs after each RAR round

target_error: 1e-3
```

### Burgers 1-D (`cases/burgers_1d/case.yaml`)

```yaml
case_name: "1D_Burgers_RAR"
pde_type:  "burgers_1d"
nu: 0.003183098861837907  # 0.01 / pi

domain:
  x_min: -1.0
  x_max:  1.0
  t_min:  0.0
  t_max:  1.0

N_col: 10000
N_bc:  200

epochs_adam:  8000
epochs_lbfgs: 1000
lr: 0.001

rar_rounds:        3
rar_points:     2000
rar_pool_size: 20000
rar_refine_epochs: 1000

target_error: 1e-2
```

---

## Adding a New PDE

1. Open `opinfd/physics.py` and subclass `BasePDE`:

```python
class HeatPDE(BasePDE):
    def residual_loss(self, model, x_col, t_col):
        ...
    def bc_ic_loss(self, model, x_ic, t_ic, x_bc, t_bc):
        ...
    def exact(self, x_np, t_np):
        ...
```

2. Register it:

```python
PDE_REGISTRY = {
    "poisson_1d": PoissonPDE,
    "burgers_1d": BurgersPDE,
    "heat_1d":    HeatPDE,     # <-- add this line
}
```

3. Add collocation sampling logic in `trainer.py → _make_collocation()` for your new `pde_type`.

4. Create `cases/heat_1d/case.yaml` with `pde_type: "heat_1d"`.

5. Run:
```bash
python -m scripts.run_case cases/heat_1d
```

---

## Architecture

```
case.yaml
    │
    ▼
run_case.py  ──►  train_case(config, case_dir)
                        │
                        ├──  get_pde(config)        → BasePDE plugin
                        ├──  SimplePINN(input_dim)  → model
                        ├──  _make_collocation()    → (x_col, t_col, ...)
                        │
                        ├──  Phase 1: Adam
                        ├──  Adaptive RAR (N rounds)
                        └──  Phase 2: L-BFGS
                                │
                                └── results/
                                      plot.png
                                      loss_history.png
                                      metrics.json
```

---

## PDEs Implemented

| `pde_type`    | Equation                                | Domain              | Exact / Reference     |
|---------------|-----------------------------------------|---------------------|-----------------------|
| `poisson_1d`  | −u_xx = π² sin(πx)                     | x ∈ [−1, 1]         | sin(πx)               |
| `burgers_1d`  | u_t + u u_x = ν u_xx, ν = 0.01/π      | x ∈ [−1,1], t∈[0,1] | FD Method-of-Lines    |

---

## Requirements

```
torch
numpy
matplotlib
pyyaml
scipy
```

---

## Citation / Acknowledgement

PINN methodology based on:
> Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
> *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.*
> Journal of Computational Physics, 378, 686–707.

---

## License

MIT © 2026 Soham Londhe

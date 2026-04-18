# Changelog

All notable changes to OPINFD are documented here.

---

## [0.2.0] — Stage 2

### Added
- **`BasePDE` plugin system** (`opinfd/physics.py`) — new PDEs subclass `BasePDE`
  and implement `residual_loss()`, `bc_ic_loss()`, and `exact()`. Trainer is
  now fully PDE-agnostic.
- **`BurgersPDE`** — 1-D viscous Burgers equation (ν = 0.01/π), IC = −sin(πx),
  zero Dirichlet BCs. Reference solution via Method-of-Lines finite differences
  (scipy). Validation produces a 3-panel heatmap (exact | PINN | |error|).
- **`PDE_REGISTRY` + `get_pde(config)`** — maps `pde_type` string in `case.yaml`
  to the correct PDE class; forwards PDE-specific kwargs (e.g. `nu`) automatically.
- **Adaptive multi-round RAR** — `rar_rounds` (default 3) consecutive enrichment
  cycles, each adding `rar_points` high-residual collocation points from a
  `rar_pool_size` candidate pool, followed by `rar_refine_epochs` Adam steps.
- **Loss history plot** (`loss_history.png`) — semi-log loss vs. iteration with
  vertical markers for Adam start, each RAR round, and L-BFGS phase.
- **`cases/burgers_1d/case.yaml`** — new benchmark case configuration.
- **`scipy`** added to `requirements.txt` (needed for Burgers FD reference).

### Changed
- **`SimplePINN`** — accepts `input_dim` (1 for steady, 2 for time-dependent).
  The `t = x` dummy-variable hack is removed. `forward(*coords)` validates the
  number of inputs at runtime.
- **`SimplePINN`** — `hidden_dim` and `n_layers` are now constructor arguments
  (defaults: 50, 2) for easy architecture sweeps.
- **`PoissonPDE`** — refactored out of `physics.py` free function into a class.
  Physics logic is identical.
- **`trainer.py`** — fully rewritten around the `BasePDE` interface.
  Collocation setup, RAR, and validation are all PDE-aware but trainer code
  contains no PDE math.
- **`cases/poisson_1d/case.yaml`** — added `pde_type`, `rar_rounds`,
  `rar_pool_size`, `rar_refine_epochs` fields.
- **`scripts/run_case.py`** — clearer error messages, prints case name.
- **`setup_opinfd.py`** — accepts optional CLI argument for case directory;
  skips conda env creation if env already exists.
- **`__version__`** bumped to `0.2.0`.

### Fixed
- `t = x` dummy variable in `SimplePINN.forward()` — steady models now
  genuinely take a single spatial input; time-dependent models take `(x, t)`.
- L-BFGS `closure` now also appends to `loss_history` for complete logging.

---

## [0.1.0] — Stage 1 (initial release)

### Added
- `SimplePINN` — 2-layer, 50-neuron Tanh network with Xavier init.
- `poisson_loss()` — free function computing PDE + BC loss for 1-D Poisson.
- `train_case()` — Adam → single RAR round → L-BFGS training loop.
- `cases/poisson_1d/case.yaml` — first benchmark case.
- `scripts/run_case.py` — CLI entry point.
- `setup_opinfd.py` — one-command conda setup and run.

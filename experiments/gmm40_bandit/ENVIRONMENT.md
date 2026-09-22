# Isolated runtimes

JAX and PyTorch use separate environments because their bundled CUDA package constraints differ. Both use Python3.11 on RTX5090. JAX0.6.2 CUDA12 + Flax0.10.4 + Optax0.2.4; install CPU Torch2.7.1 first for SB3 and target initialization. PyTorch baselines use Torch2.7.1 **cu128** (Blackwell kernels), not cu124. Install each requirements-*.txt afterwards. Capture `uv pip freeze` separately in the deployment record.

Set `GMM40_PYTHON` to the JAX interpreter and `GMM40_TORCH_PYTHON` to the Torch interpreter. Both run from the identical committed code snapshot. The launcher uses one subprocess per run, one worker per visible GPU. No ports or external services are opened. Use tmux as requested in the experiment workflow.

CPU unit validation (2026-09-22): Python3.12/JAX0.6.2/Flax0.10.4/Optax0.2.4/Torch2.7.1; 12 numerical checks passed before GPU preflight. This is not a GPU-speed measurement. GPU package freezes and preflight measurements are authoritative for experiment execution.

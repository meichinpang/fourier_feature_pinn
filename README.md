# Fourier Feature Hybrid PINN for a Damped Pendulum

This repository implements a **Hybrid Physics-Informed Neural Network (PINN)** that combines ODE-based physics constraints with noisy, sparse and missing experimental observations to solve the damped pendulum equation:

$$
\boxed{\frac{d^2\theta}{dt^2} + \frac{b}{m}\frac{d\theta}{dt} + \frac{g}{L}\sin\bigl(\theta(t)\bigr) = 0}
$$

| Parameter                  | Symbol           |         Value | Unit  |
| -------------------------- | ---------------- | ------------: | ----- |
| Gravitational acceleration | $g$              |          9.81 | m/s²  |
| Pendulum length            | $L$              |           1.0 | m     |
| Mass                       | $m$              |           1.0 | kg    |
| Damping coefficient        | $b$              |          0.25 | kg/s  |
| Initial angle              | $\theta_0$       | $\pi/4$ (45°) | rad   |
| Initial angular velocity   | $\omega_0$       |           0.0 | rad/s |
| Start time                 | $t_\text{start}$ |             0 | s     |
| End time                   | $t_\text{end}$   |            20 | s     |

Two architectures are compared:

| Notebook                        | Architecture                                    | Key idea                                                                                       |
| ------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `01_baseline_hybrid_pinn.ipynb` | Fully connected (`Tanh`)                        | Standard MLP baseline                                                                          |
| `02_fourier_feature_pinn.ipynb` | Fourier feature embedding + curriculum training | Overcomes spectral bias with physics-informed frequency bands and staged time-window expansion |

## Prerequisites

### Install `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. Install it with:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Create the virtual environment and install dependencies

```bash
uv sync --group dev
```

This reads `pyproject.toml` and installs all runtime and dev dependencies (including `ipykernel` and `notebook`).

### GPU setup: CUDA vs XPU

PyTorch must be installed with the correct backend for your GPU hardware. The default `uv sync` installs a CPU-only or CUDA build. Choose **one** of the options below.

#### Option A: NVIDIA GPU (CUDA)

```bash
uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
```

In the notebook, set:

```python
gpu_type = "cuda"
```

#### Option B: Intel GPU (XPU)

Intel GPUs (Iris Xe, Arc, Data Center Max) require the **XPU build** of PyTorch and the Intel Extension for PyTorch runtime libraries.

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu
```

In the notebook, set:

```python
gpu_type = "xpu"
```

> **Note:** The standard CUDA wheel (`torch+cu130`) ships with `USE_XPU=OFF`, so `torch.xpu.is_available()` will always return `False` even if Intel runtime libraries are installed. You must install from Intel's wheel index to get a build compiled with `USE_XPU=ON`.

#### Verify GPU detection

```python
import torch

# CUDA
print(torch.cuda.is_available())       # True if NVIDIA driver + CUDA build
print(torch.cuda.get_device_name(0))

# XPU
print(torch.xpu.is_available())        # True if Intel driver + XPU build
print(torch.xpu.get_device_name(0))
```

## Experiment Scenarios

Four scenarios test the PINN under increasingly challenging conditions:

| Scenario | Description                          | Data points | Time range  | Noise $\sigma$ (rad) | Challenge                                                |
| :------: | ------------------------------------ | :---------: | :---------: | :------------------: | -------------------------------------------------------- |
|    1     | Clean experimental data              |     250     | $[0, 20]$ s |        0.001         | Baseline: low-noise, dense measurements                  |
|    2     | Noisy experimental data              |     250     | $[0, 20]$ s |         0.1          | High observation noise obscures the true signal          |
|    3     | Sparse experimental data             |     75      | $[0, 20]$ s |        0.001         | Few data points to constrain the solution                |
|    4     | Extrapolation beyond training window |     75      | $[0, 10]$ s |        0.001         | PINN must predict $t \in [10, 20]$ s using physics alone |

## Baseline PINN Hyperparameters

The baseline uses a fully connected network with `Tanh` activations. Only Scenarios 1 and 2 are tested because the baseline already degrades significantly under noise.

| Hyperparameter        |        Scenario 1 (Clean) |         Scenario 2 (Noisy) |
| --------------------- | ------------------------: | -------------------------: |
| Hidden units          |                        64 |                         64 |
| Hidden layers         |                         3 |                          4 |
| Collocation points    |                     1,000 |                      2,000 |
| Epochs                |                    10,000 |                     80,000 |
| Learning rate         |                 $10^{-3}$ |                  $10^{-3}$ |
| LR scheduler          | StepLR, halve every 5,000 | StepLR, halve every 20,000 |
| $\lambda_\text{IC}$   |                         1 |                          5 |
| $\lambda_\text{phys}$ |                         1 |                         10 |
| $\lambda_\text{data}$ |                         1 |                          1 |
| Optimizer             |                      Adam |                       Adam |

## Hybrid Fourier Feature PINN Hyperparameters

The Fourier feature PINN replaces the first linear layer with a frozen sinusoidal embedding using five physics-informed frequency bands derived from the natural frequency $\omega_n = \sqrt{g/L} \approx 3.13$ rad/s:

| Band                          |                          Frequency |
| ----------------------------- | ---------------------------------: |
| Sub-harmonic (decay envelope) | $0.5\,\omega_n \approx 1.57$ rad/s |
| Fundamental                   | $1.0\,\omega_n \approx 3.13$ rad/s |
| Nonlinear frequency shift     | $1.5\,\omega_n \approx 4.71$ rad/s |
| 2nd harmonic                  | $2.0\,\omega_n \approx 6.26$ rad/s |
| 3rd harmonic                  | $3.0\,\omega_n \approx 9.39$ rad/s |

The input dimension becomes $1 + 2K = 11$ (1 normalized time + 5 sin/cos pairs). All scenarios use the same architecture and curriculum schedule; only the loss weights differ.

### Architecture (all scenarios)

| Parameter                 |                           Value |
| ------------------------- | ------------------------------: |
| Input dimension           | 11 (1 + 2 × 5 Fourier features) |
| Hidden units              |                              64 |
| Hidden layers             |                               4 |
| Activation                |                            Tanh |
| Optimizer                 |                            Adam |
| Learning rate             |                       $10^{-3}$ |
| Data loss                 |          Huber ($\delta = 0.5$) |
| Causal weight $\alpha$    |                             3.0 |
| Exponential decay $\beta$ |                             2.0 |

### Curriculum training schedule (all scenarios)

| Stage | Time window | Epochs | Collocation points | Cumulative epochs |
| :---: | :---------: | -----: | -----------------: | ----------------: |
|   1   | $[0, 5]$ s  |  2,000 |                300 |             2,000 |
|   2   | $[0, 10]$ s |  3,000 |                600 |             5,000 |
|   3   | $[0, 15]$ s |  4,000 |                900 |             9,000 |
|   4   | $[0, 20]$ s |  8,000 |              1,200 |            17,000 |

### Loss weights per scenario

| Loss weight           | Scenario 1 (Clean) | Scenario 2 (Noisy) | Scenario 3 (Sparse) | Scenario 4 (Extrapolation) |
| --------------------- | -----------------: | -----------------: | ------------------: | -------------------------: |
| $\lambda_\text{IC}$   |                  1 |                  1 |                   1 |                          1 |
| $\lambda_\text{phys}$ |                  1 |                 10 |                   1 |                         10 |
| $\lambda_\text{data}$ |                  1 |                  1 |                  10 |                          1 |

Key tuning rationale for each scenario:

* Due to the increase in the noise level ($\sigma = 0.1\;\text{rad}$) in Scenario (2), we increase the physics loss weight to 10 ($\lambda_\text{phys} = 10$) so that physics loss becomes more dominant to prevent fitting noisy scatter points. 
* In Scenario (3), we can trust the measurements by increasing data loss weight to 10 ($\lambda_\text{data} = 10$) as the data points are sparse but trustworthy ($\sigma = 0.001\;\text{rad}$). 
* In Scenario (4) where data covers only [0, 10] s (first ~5 oscillations), the PINN must predict [10, 20] s purely from physics. Therefore, we raise $\lambda_\text{phys}$ to 10 because there are zero data anchors beyond the first five oscillations. Here, the data-free region is entirely physics-driven due to available collocation points, where the ODE residual is the only constraint keeping the solution on track.

## References

1. Raissi, M., Perdikaris, P. & Karniadakis, G. E. "Physics-informed neural networks." *J. Comput. Phys.* 378 (2019). [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
2. Wang, S., Sankaran, S., Wang, H. & Perdikaris, P. "An expert's guide to training physics-informed neural networks." *arXiv:2308.08468* (2023). [arXiv](https://arxiv.org/abs/2308.08468)
3. Francis Fernandes (2026). [Mastering Dynamics PINNs](https://topmate.io/pinnsformechanicalengineers).
4. Dao, D. L. "Experimental evaluation of damping models for a nonlinear pendulum system." *Phys. Educ.* 58 (2023). [doi:10.1088/1361-6552/ace1ca](https://doi.org/10.1088/1361-6552/ace1ca)

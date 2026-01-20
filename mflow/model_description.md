# Multi-Fidelity Kriging Model: Mathematical Theory and Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Framework](#mathematical-framework)
3. [Gaussian Process Regression Fundamentals](#gaussian-process-regression-fundamentals)
4. [Multi-Fidelity Co-Kriging](#multi-fidelity-co-kriging)
5. [Kernel Design and Selection](#kernel-design-and-selection)
6. [Training Procedure](#training-procedure)
7. [Prediction Algorithm](#prediction-algorithm)
8. [Hyperparameter Optimization](#hyperparameter-optimization)
9. [Uncertainty Quantification](#uncertainty-quantification)
10. [NaN Handling in Building Regions](#nan-handling-in-building-regions)
11. [Computational Considerations](#computational-considerations)
12. [Implementation Details](#implementation-details)

---

## Introduction

This document provides a comprehensive mathematical description of the multi-fidelity kriging model used for wind field prediction. The model combines:

- **Low-fidelity data**: RANS (Reynolds-Averaged Navier-Stokes) simulations at 360 angles
- **High-fidelity data**: LES (Large Eddy Simulation) at 48 specific collocation points

The goal is to predict wind velocity fields at arbitrary angles and spatial locations with LES-level accuracy while leveraging the comprehensive angular coverage of RANS data.

---

## Mathematical Framework

### Problem Statement

Given:
- Low-fidelity observations: **y<sub>LF</sub>** at locations **X<sub>LF</sub>**
- High-fidelity observations: **y<sub>HF</sub>** at locations **X<sub>HF</sub>**
- Input space: **x** = (θ, x, y) where θ is wind angle, (x, y) are spatial coordinates

Predict:
- High-fidelity value **ŷ<sub>HF</sub>**(x\*) at any test point x\*

### Multi-Fidelity Relationship

The fundamental assumption is an **autoregressive structure**:

```
y_HF(x) = ρ · y_LF(x) + δ(x)
```

where:
- **ρ** ∈ ℝ: Scaling factor (constant or spatially varying)
- **δ(x)**: Residual/correction term modeled as a Gaussian Process
- **y_LF(x)**: Low-fidelity prediction (RANS)
- **y_HF(x)**: High-fidelity truth (LES)

### Physical Interpretation

1. **ρ · y_LF(x)**: RANS captures the mean flow structure but with systematic bias
2. **δ(x)**: Correction term capturing:
   - Turbulent fluctuations RANS misses
   - Wall effects and boundary layer details
   - Separation and recirculation zones
   - Small-scale flow features

---

## Gaussian Process Regression Fundamentals

### Definition

A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. It is fully specified by:

- **Mean function**: m(x) = 𝔼[f(x)]
- **Covariance function** (kernel): k(x, x') = 𝔼[(f(x) - m(x))(f(x') - m(x'))]

### Prior Distribution

```
f(x) ~ GP(m(x), k(x, x'))
```

Common choices:
- **Mean**: m(x) = 0 (zero mean) or m(x) = μ (constant mean)
- **Kernel**: RBF, Matérn, etc. (see [Kernel Design](#kernel-design-and-selection))

### Posterior Distribution

Given training data **D** = {(**X**, **y**)}, the posterior predictive distribution at test point x\* is:

```
f(x*) | D ~ N(μ_posterior(x*), σ²_posterior(x*))
```

where:

**Posterior mean**:
```
μ_posterior(x*) = k(x*, X) [K + σ_n² I]⁻¹ y
```

**Posterior variance**:
```
σ²_posterior(x*) = k(x*, x*) - k(x*, X) [K + σ_n² I]⁻¹ k(X, x*)
```

where:
- **K**: N×N covariance matrix, K<sub>ij</sub> = k(x<sub>i</sub>, x<sub>j</sub>)
- **k(x\*, X)**: 1×N vector of covariances between x\* and training points
- **σ<sub>n</sub>²**: Noise variance (observation noise)
- **I**: Identity matrix

### Key Properties

1. **Non-parametric**: Flexibility grows with data
2. **Uncertainty quantification**: Natural confidence intervals
3. **Smoothness**: Controlled by kernel choice
4. **Interpolation**: Exact fit at training points (if σ<sub>n</sub> → 0)

---

## Multi-Fidelity Co-Kriging

### Autoregressive Co-Kriging Model

Our implementation uses the **autoregressive (AR1) co-kriging** approach:

**Step 1: Model the low-fidelity data**
```
y_LF(x) ~ GP(m_LF(x), k_LF(x, x'))
```

**Step 2: Model the discrepancy**
```
δ(x) = y_HF(x) - ρ · y_LF(x)
δ(x) ~ GP(0, k_δ(x, x'))
```

**Step 3: Multi-fidelity prediction**
```
y_HF(x*) = ρ · y_LF(x*) + δ(x*)
```

### Advantages of AR1 Co-Kriging

1. **Computational efficiency**: Only need to fit one GP (for δ) instead of coupled GPs
2. **Leverages correlation**: ρ captures systematic relationship between fidelities
3. **Improved predictions**: Combines structural information from RANS with LES accuracy
4. **Reduced uncertainty**: LF data provides information away from HF observations

### Estimation of Scaling Factor ρ

We estimate ρ using **least squares** on the high-fidelity training data:

```
ρ = argmin_ρ Σᵢ [y_HF(xᵢ) - ρ · y_LF(xᵢ)]²
```

Closed-form solution:
```
ρ = (y_HF^T y_LF) / (y_LF^T y_LF)
```

where y_LF is evaluated at high-fidelity locations via interpolation.

**Physical interpretation**: ρ represents the average ratio between LES and RANS velocities, typically ρ ≈ 1.0 to 1.2 due to RANS underprediction of peak velocities.

---

## Kernel Design and Selection

The covariance kernel k(x, x') encodes assumptions about the smoothness and structure of the residual field δ(x).

### Radial Basis Function (RBF) Kernel

**Isotropic RBF**:
```
k_RBF(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
```

**Anisotropic RBF** (Automatic Relevance Determination):
```
k_ARD(x, x') = σ² exp(-Σᵢ (xᵢ - x'ᵢ)² / (2ℓᵢ²))
```

where:
- **σ²**: Signal variance (output scale)
- **ℓ** or **ℓᵢ**: Length scale(s)
- **||·||**: Euclidean distance

**Properties**:
- Infinitely differentiable (smooth functions)
- Stationary (translation invariant)
- Universal approximator

### Matérn Kernel (Advanced Model)

```
k_Matérn(x, x') = σ² · (2^(1-ν) / Γ(ν)) · (√(2ν) r / ℓ)^ν · K_ν(√(2ν) r / ℓ)
```

where:
- **r** = ||x - x'||
- **ν**: Smoothness parameter (ν = 1/2, 3/2, 5/2, ...)
- **K<sub>ν</sub>**: Modified Bessel function
- **Γ**: Gamma function

**Properties**:
- Matérn 1/2: Equivalent to exponential kernel (continuous but not differentiable)
- Matérn 3/2: Once differentiable
- Matérn 5/2: Twice differentiable
- Matérn ∞: Equivalent to RBF (infinitely differentiable)

**Advantage over RBF**: More flexible, can model rougher functions

### Anisotropic Length Scales

In our 3D input space **x** = (θ, x_spatial, y_spatial), we use different length scales:

```
k(x, x') = σ² exp(-(θ - θ')² / (2ℓ_angle²)
                   -(x - x')² / (2ℓ_x²)
                   -(y - y')² / (2ℓ_y²))
```

**Typical values**:
- **ℓ<sub>angle</sub>** ≈ 10° (wind patterns vary smoothly with angle)
- **ℓ<sub>spatial</sub>** ≈ 50 grid units (spatial correlation length)

**Rationale**: Wind fields are smoother in angle than in space (buildings create sharp spatial gradients).

### White Noise Kernel

```
k_noise(x, x') = σ_n² δ(x - x')
```

where δ is the Kronecker delta.

**Purpose**:
- Models observation noise
- Ensures numerical stability (matrix K + σ<sub>n</sub>²I is positive definite)
- Prevents overfitting

### Composite Kernel

The final kernel is typically a sum:

```
k_total(x, x') = k_RBF(x, x') + k_noise(x, x')
```

or in matrix form:
```
K_total = K_RBF + σ_n² I
```

---

## Training Procedure

### Overview

The training phase estimates:
1. Scaling factor **ρ**
2. GP hyperparameters **θ** = {σ², ℓ, σ<sub>n</sub>²}

### Step-by-Step Algorithm

**Input**: High-fidelity data {**X<sub>HF</sub>**, **y<sub>HF</sub>**}, low-fidelity model y<sub>LF</sub>(·)

**Step 1: Data Preparation**
```
For each HF point xᵢ ∈ X_HF:
    Evaluate y_LF(xᵢ) via interpolation of RANS data
```

**Step 2: Estimate Scaling Factor**
```
Filter valid points: idx = {i : y_HF(xᵢ) > ε and y_LF(xᵢ) > ε and isfinite(...)}
ρ = Σᵢ∈idx [y_HF(xᵢ) · y_LF(xᵢ)] / Σᵢ∈idx [y_LF(xᵢ)²]
```

**Step 3: Compute Residuals**
```
δᵢ = y_HF(xᵢ) - ρ · y_LF(xᵢ)  for i = 1, ..., N
```

**Step 4: Fit GP to Residuals**
```
Maximize marginal log-likelihood:
    θ* = argmax_θ log p(δ | X_HF, θ)
```

where:
```
log p(δ | X, θ) = -½ δ^T K⁻¹ δ - ½ log|K| - N/2 log(2π)
```

**Step 5: Store Model**
```
Save: ρ, θ*, K⁻¹, X_HF, δ
```

### NaN Filtering (Important for Building Regions)

Before training, filter out invalid points:

```python
# Filter NaN values from building regions
valid_mask = (np.isfinite(y_hf) &
              np.isfinite(y_lf) &
              (y_hf > 1e-6) &
              (y_lf > 1e-6))

X_hf = X_hf[valid_mask]
y_hf = y_hf[valid_mask]
y_lf = y_lf[valid_mask]
```

This ensures:
- No NaN values in training data
- Finite residuals δ
- Numerically stable optimization

### Hyperparameter Optimization

**Objective**: Maximize marginal log-likelihood

```
L(θ) = log p(δ | X, θ) = -½ δ^T K⁻¹ δ - ½ log|K| - N/2 log(2π)
```

**Gradient**:
```
∂L/∂θⱼ = ½ trace[(δδ^T K⁻¹ - K⁻¹) ∂K/∂θⱼ]
```

**Optimizer**:
- **CPU version**: L-BFGS-B (scipy.optimize)
- **GPU version**: Adam optimizer (torch.optim)

**Iterations**:
- **CPU**: 3-5 random restarts
- **GPU**: 50-5000 iterations depending on data size

---

## Prediction Algorithm

### Single Point Prediction

**Input**: Test point x\*, trained model {ρ, θ\*, K<sup>-1</sup>, **X<sub>HF</sub>**, **δ**}

**Step 1: Get Low-Fidelity Prediction**
```
y_LF(x*) = Interpolate RANS at (θ*, x*, y*)
```

For angular interpolation (RANS at 1° resolution):
```
θ_low = ⌊θ*⌋
θ_high = ⌈θ*⌉
α = θ* - θ_low
y_LF(x*) = (1-α) · RANS[θ_low] + α · RANS[θ_high]
```

Then spatial interpolation (linear/cubic).

**Step 2: Compute Kernel Vector**
```
k* = [k(x*, x₁), k(x*, x₂), ..., k(x*, xₙ)]^T
```

**Step 3: GP Prediction of Residual**
```
δ*(x*) = k*^T K⁻¹ δ
```

**Step 4: Multi-Fidelity Combination**
```
ŷ_HF(x*) = ρ · y_LF(x*) + δ*(x*)
```

**Step 5: Uncertainty Quantification**
```
Var[ŷ_HF(x*)] = k(x*, x*) - k*^T K⁻¹ k*
σ(x*) = √Var[ŷ_HF(x*)]
```

### Field Prediction

To predict entire wind field at angle θ\*:

```
For each grid point (xᵢ, yⱼ):
    x* = (θ*, xᵢ, yⱼ)
    ŷ[i,j], σ[i,j] = Predict(x*)
```

**Optimization**: Batch processing on GPU
```python
# Create batch of all grid points
X_test = [(θ*, xᵢ, yⱼ) for i in range(nx) for j in range(ny)]

# Batch prediction (vectorized)
ŷ, σ = model.predict(X_test, data, return_std=True)

# Reshape to grid
prediction = ŷ.reshape(nx, ny)
uncertainty = σ.reshape(nx, ny)
```

---

## Hyperparameter Optimization

### Marginal Log-Likelihood

The **marginal likelihood** integrates out the function values:

```
p(y | X, θ) = ∫ p(y | f, X) p(f | X, θ) df
             = N(y | 0, K_θ + σ_n² I)
```

**Log marginal likelihood**:
```
log p(y | X, θ) = -½ y^T (K + σ_n² I)⁻¹ y
                  - ½ log|K + σ_n² I|
                  - N/2 log(2π)
```

**Three terms**:
1. **Data fit**: -½ y<sup>T</sup> K<sup>-1</sup> y (reward fit)
2. **Complexity penalty**: -½ log|K| (penalize complex models)
3. **Normalization constant**: -N/2 log(2π)

### Optimization Strategy

**Bounds on hyperparameters**:
```
σ² ∈ [1e-3, 1e3]     # Signal variance
ℓ ∈ [1.0, 100.0]     # Length scale(s)
σ_n² ∈ [1e-6, 1e-1]  # Noise variance
```

**Multi-start optimization**:
```
For restart = 1 to n_restarts:
    θ₀ ~ Random initialization within bounds
    θ* = Optimize L(θ) starting from θ₀
    Store θ* and L(θ*)
Return θ* with highest L(θ*)
```

**Priors** (Bayesian approach):
```
ℓ ~ Gamma(3.0, 6.0)  # Weakly informative prior
```

---

## Uncertainty Quantification

### Sources of Uncertainty

1. **Model uncertainty**: GP posterior variance
2. **Low-fidelity uncertainty**: RANS interpolation error
3. **Parameter uncertainty**: Hyperparameter estimation error (not included)

### Predictive Variance

```
Var[ŷ(x*)] = σ²_δ(x*) + 0  (ignoring LF uncertainty)
```

where:
```
σ²_δ(x*) = k(x*, x*) - k(x*, X)^T [K + σ_n² I]⁻¹ k(x*, X)
```

**Interpretation**:
- **k(x\*, x\*)**: Prior variance (maximum possible)
- Subtraction term: Information gain from data
- σ<sub>δ</sub>(x\*) is small near training points, large far away

### Confidence Intervals

95% confidence interval:
```
[ŷ(x*) - 1.96·σ(x*), ŷ(x*) + 1.96·σ(x*)]
```

### Expected Improvement (EI)

For adaptive sampling:
```
EI(x*) = σ(x*) [z·Φ(z) + φ(z)]
```

where:
- z = (ŷ(x\*) - y<sub>best</sub>) / σ(x\*)
- Φ, φ: Standard normal CDF and PDF

---

## NaN Handling in Building Regions

### Problem

Wind data contains **NaN values** in building/obstacle regions where:
- Flow velocity is undefined (inside solid objects)
- Simulation domain is masked

### Solution Strategy

**1. Data Preparation Filter**
```python
# Only include valid flow regions
if (np.isfinite(les_val) and np.isfinite(rans_val) and
    les_val > 1e-6 and rans_val > 1e-6):
    Include in training set
```

**2. Training-Time Validation**
```python
# Filter any remaining NaN
valid_mask = (np.isfinite(y_hf) &
              np.isfinite(y_lf) &
              (y_lf > 1e-6))

if valid_mask.sum() < len(y_hf):
    print(f"Filtered {len(y_hf) - valid_mask.sum()} NaN points")
    y_hf = y_hf[valid_mask]
    # ... filter X_hf, y_lf similarly
```

**3. Prediction-Time Handling**
```python
# Set predictions to 0 in building regions
nan_mask = ~np.isfinite(y_lf_test)
if nan_mask.any():
    y_pred[nan_mask] = 0.0
    uncertainty[nan_mask] = 0.0
```

### Physical Justification

- **Inside buildings**: Velocity = 0 (no-slip condition at walls)
- **GP doesn't train there**: Model has no information about building interiors
- **Set to 0**: Physically correct and prevents NaN propagation

---

## Computational Considerations

### Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Training | O(N³) | Cholesky decomposition of K |
| Prediction (single) | O(N²) | Matrix-vector product |
| Prediction (M points) | O(M·N²) | Can be batched |
| Hyperparameter opt. | O(I·N³) | I iterations |

where N = number of training points.

### Memory Requirements

**CPU version**:
- **K matrix**: N² × 8 bytes (float64)
- Example: N=50,000 → 20 GB

**GPU version**:
- **K matrix**: N² × 4 bytes (float32)
- Example: N=50,000 → 10 GB

### Scaling Strategies

**1. Subsampling**
```python
n_sample = int(len(y_hf) * subsample_ratio)
indices = np.random.choice(len(y_hf), n_sample, replace=False)
X_train = X_hf[indices]
```

**2. Batch Training (GPU)**
```python
# Train on random batches iteratively
for iter in range(n_iterations):
    batch_idx = random_choice(n_samples, batch_size)
    X_batch = X_train[batch_idx]
    y_batch = y_train[batch_idx]
    # Update model with batch
```

**3. Inducing Points (Sparse GP)**
- Select M << N inducing points
- Complexity: O(M²N)
- Not currently implemented

**4. GPU Acceleration**
- 10-100× speedup for training
- Batch prediction on GPU
- Requires: PyTorch, GPyTorch

### Numerical Stability

**1. Jitter Addition**
```
K_stable = K + (σ_n² + ε) I
```
where ε = 1e-6 (jitter term).

**2. Cholesky Decomposition**
```python
try:
    L = cholesky(K)
except LinAlgError:
    # Add jitter and retry
    K += 1e-6 * np.eye(N)
    L = cholesky(K)
```

**3. Constraint Enforcement**
```
σ_n² ≥ 1e-6  (noise floor)
ℓ ≥ 1.0       (minimum length scale)
```

---

## Implementation Details

### CPU Implementation (scikit-learn)

**Class**: `MultiFidelityGPR` ([multifidelity_gpr.py](multifidelity_gpr.py))

**Key components**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=[ℓ, ℓ, ℓ]) + WhiteKernel(σ_n²)
gp = GaussianProcessRegressor(kernel=kernel,
                                n_restarts_optimizer=3,
                                normalize_y=True)
gp.fit(X_hf, delta)
```

**Advantages**:
- Simple, well-tested
- Automatic hyperparameter optimization
- Good for small datasets (N < 10,000)

**Limitations**:
- Slow for large N
- No GPU support

### GPU Implementation (GPyTorch)

**Class**: `MultiFidelityGPR_GPU` ([multifidelity_gpr_gpu.py](multifidelity_gpr_gpu.py))

**Key components**:
```python
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, length_scale):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )
```

**Training loop**:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
```

**Advantages**:
- GPU acceleration (10-100× faster)
- Handles large datasets (N > 100,000)
- Batch training mode
- Automatic differentiation

### Advanced Model

**Class**: `AdvancedMultiFidelityGPR_GPU` ([advanced_multifidelity_gpr_gpu.py](advanced_multifidelity_gpr_gpu.py))

**Additional features**:
- Matérn kernel option
- Anisotropic length scales (different for angle vs. space)
- Cross-validation
- Comprehensive diagnostics

### Data Pipeline

**Input**:
- RANS: 360 files × (nx, ny) grid
- LES: 48 files × (nx, ny) grid

**Processing**:
```python
# 1. Load data
x_grid, y_grid, data = model.load_data('truncated_data')

# 2. Prepare training pairs
X_hf, y_hf, y_lf = model.prepare_training_data(data)
# X_hf: (N, 3) array of (angle, x, y)
# y_hf: (N,) array of LES velocities
# y_lf: (N,) array of RANS velocities at HF locations

# 3. Filter NaN
valid_mask = np.isfinite(y_hf) & np.isfinite(y_lf) & (y_lf > 1e-6)
X_hf, y_hf, y_lf = X_hf[valid_mask], y_hf[valid_mask], y_lf[valid_mask]

# 4. Train
model.fit(X_hf, y_hf, y_lf, training_iter=50)

# 5. Predict
prediction, uncertainty = model.predict_field(angle=45.0, data=data)
```

---

## Summary

The multi-fidelity kriging model implemented here:

1. **Leverages two data sources**: Combines extensive RANS coverage with sparse but accurate LES data

2. **Uses autoregressive structure**: y<sub>HF</sub> = ρ·y<sub>LF</sub> + δ, where δ is a GP

3. **Provides uncertainty**: Gaussian process gives predictive variance

4. **Handles NaN robustly**: Filters building regions at multiple stages

5. **Scales to large data**: GPU acceleration and batch training

6. **Physically informed**: Kernel design reflects flow physics (smooth in angle, structured in space)

### Key Equations

**Training**:
```
ρ = (y_HF^T y_LF) / (y_LF^T y_LF)
δ = y_HF - ρ · y_LF
θ* = argmax_θ log p(δ | X_HF, θ)
```

**Prediction**:
```
ŷ_HF(x*) = ρ · y_LF(x*) + k(x*, X)^T [K + σ_n² I]⁻¹ δ
σ²(x*) = k(x*, x*) - k(x*, X)^T [K + σ_n² I]⁻¹ k(x*, X)
```

**Kernel (Anisotropic RBF)**:
```
k(x, x') = σ² exp(-Σᵢ (xᵢ - x'ᵢ)² / (2ℓᵢ²))
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-19
**Author**: Multi-Fidelity GPR Development Team

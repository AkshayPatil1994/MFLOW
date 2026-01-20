import numpy as np
import torch
import gpytorch
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import warnings
import pickle
warnings.filterwarnings('ignore')
#
# DEFINE CLASS FOR GPyTorch MODEL
#
class ExactGPModel(gpytorch.models.ExactGP):
    """
        GPyTorch Exact GP model for the residual delta.
    """
    def __init__(self, train_x, train_y, likelihood, kernel_length_scale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # Anisotropic RBF kernel with different length scales for [angle, x, y]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=3,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            )
        )

        # Initialize length scales
        self.covar_module.base_kernel.lengthscale = torch.tensor(
            [[kernel_length_scale, kernel_length_scale, kernel_length_scale]]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
# GPU-ACCELERATED MULTI-FIDELITY GPR CLASS
#
class MultiFidelityGPR_GPU:
    """
        GPU-Accelerated Multi-Fidelity Gaussian Process Regression for wind field prediction.

        Implements the co-kriging approach:
        y_high(x) = rho * y_low(x) + delta(x)

        where:
        - y_high: High-fidelity (LES) data
        - y_low: Low-fidelity (RANS) data
        - rho: Scaling coefficient
        - delta: GP-based correction term (trained on GPU)

        Parameters
        ----------
        kernel_length_scale : float
            Initial length scale for RBF kernel (controls smoothness)
        noise_level : float
            White noise level for numerical stability
        device : str
            Device to use: 'cuda', 'cpu', or 'auto' (default: 'auto')
    """

    def __init__(self, kernel_length_scale: float = 10.0,
                 noise_level: float = 0.01,
                 device: str = 'auto',
                 max_training_samples: int = None):
        """
            Initialize the GPU-accelerated multi-fidelity GPR model.

            Parameters
            ----------
            max_training_samples : int, optional
                Maximum number of training samples to prevent OOM.
                If None, automatically determined based on available GPU memory.
        """
        self.kernel_length_scale = kernel_length_scale
        self.noise_level = noise_level

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {total_mem:.2f} GB")

            # Set maximum training samples based on GPU memory
            if max_training_samples is None:
                # Conservative estimate: ~5000 samples per GB
                self.max_training_samples = int(total_mem * 5000)
                print(f"Auto-set max training samples: {self.max_training_samples:,}")
            else:
                self.max_training_samples = max_training_samples
        else:
            # CPU mode - can handle more samples but slower
            self.max_training_samples = max_training_samples if max_training_samples else 100000

        self.gp_model = None
        self.likelihood = None
        self.rho = None
        self.X_train_hf = None
        self.y_train_hf = None
        self.y_train_lf_at_hf = None

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_gpu_memory_info(self):
        """Get current GPU memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - allocated
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total
            }
        return None

    def print_memory_usage(self, prefix=""):
        """Print current GPU memory usage."""
        mem_info = self.get_gpu_memory_info()
        if mem_info:
            print(f"{prefix}GPU Memory: {mem_info['allocated']:.2f}GB allocated, "
                  f"{mem_info['free']:.2f}GB free, {mem_info['total']:.2f}GB total")
    #
    # This method needs updating to load your non-standard data input!!
    #
    def load_data(self, data_dir: str = 'truncated_data') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
            Load RANS and LES data from files.

            Parameters
            ----------
            data_dir : str
                Directory containing the data files

            Returns
            -------
            x_grid : np.ndarray
                X coordinates of the grid
            y_grid : np.ndarray
                Y coordinates of the grid
            data : dict
                Dictionary containing 'rans' and 'les' data
        """
        # Load spatial grid (common for LES and RANS)
        x_grid = np.loadtxt(f'{data_dir}/x_common_grid.txt')
        y_grid = np.loadtxt(f'{data_dir}/y_common_grid.txt')

        # LES collocation points and experiment numbers
        les_angles = np.array([0.59, 4.37, 16.57, 37.28, 51.62, 57.55, 61.7, 65.28,
                               69.16, 73.95, 80.24, 88.15, 97.2, 105.51, 110.89, 114.42,
                               117.54, 121.19, 126.45, 133.31, 139.9, 149.34, 163.13, 172.08,
                               177.08, 180.76, 185.65, 192.73, 203.04, 216.59, 237.0, 261.88,
                               275.8, 283.3, 290.15, 298.0, 310.01, 325.46, 336.53, 341.85,
                               345.11, 347.46, 349.42, 351.3, 353.16, 355.14, 357.14, 358.86])

        exp_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

        # Load LES data (high-fidelity)
        les_data = []
        for exp_num in exp_numbers:
            filename = f'{data_dir}/U_les_normalised_{exp_num:03d}.txt'
            data_field = np.loadtxt(filename, skiprows=1)
            les_data.append(data_field)
        les_data = np.array(les_data)  # Shape: (n_angles, nx, ny)

        # Load RANS data (low-fidelity) - 1 degree resolution
        rans_angles = np.arange(1, 361, 1)  # 1 to 360 degrees
        rans_data = []
        for angle in rans_angles:
            filename = f'{data_dir}/U_rans_normalised_{angle:03d}.txt'
            data_field = np.loadtxt(filename, skiprows=1)
            rans_data.append(data_field)
        rans_data = np.array(rans_data)  # Shape: (360, nx, ny)

        data = {
            'les': {'angles': les_angles, 'data': les_data},
            'rans': {'angles': rans_angles, 'data': rans_data},
            'grid': {'x': x_grid, 'y': y_grid}
        }

        print(f"Loaded data:")
        print(f"  LES: {les_data.shape[0]} angles, grid: {les_data.shape[1]}x{les_data.shape[2]}")
        print(f"  RANS: {rans_data.shape[0]} angles, grid: {rans_data.shape[1]}x{rans_data.shape[2]}")

        return x_grid, y_grid, data

    def prepare_training_data(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Prepare training data for multi-fidelity GP.

            Parameters
            ----------
            data : dict
                Dictionary containing LES and RANS data

            Returns
            -------
            X_hf : np.ndarray
                High-fidelity input points (angle, x, y)
            y_hf : np.ndarray
                High-fidelity output (LES wind values)
            y_lf_at_hf : np.ndarray
                Low-fidelity output at high-fidelity points (interpolated RANS)
        """
        les_angles = data['les']['angles']
        les_data = data['les']['data']
        rans_angles = data['rans']['angles']
        rans_data = data['rans']['data']
        x_grid = data['grid']['x']
        y_grid = data['grid']['y']

        n_angles = len(les_angles)
        nx, ny = len(x_grid), len(y_grid)

        # Create meshgrid for spatial coordinates
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

        # Prepare high-fidelity training data
        X_hf = []
        y_hf = []
        y_lf_at_hf = []

        for i, angle in enumerate(les_angles):
            # LES data at this angle
            les_field = les_data[i]  # Shape: (nx, ny)

            # Interpolate RANS to this angle (RANS has 1-degree resolution)
            angle_idx_low = int(np.floor(angle)) % 360
            angle_idx_high = int(np.ceil(angle)) % 360
            alpha = angle - np.floor(angle)

            rans_field = (1 - alpha) * rans_data[angle_idx_low] + alpha * rans_data[angle_idx_high]

            # Flatten and create input-output pairs
            for ix in range(nx):
                for iy in range(ny):
                    # Skip NaN values (building regions) and zero/small values (masked regions)
                    les_val = les_field[ix, iy]
                    rans_val = rans_field[ix, iy]

                    # Only include points where both LES and RANS are valid (not NaN) and non-zero
                    if (np.isfinite(les_val) and np.isfinite(rans_val) and
                        les_val > 1e-6 and rans_val > 1e-6):
                        X_hf.append([angle, X_mesh[ix, iy], Y_mesh[ix, iy]])
                        y_hf.append(les_val)
                        y_lf_at_hf.append(rans_val)

        X_hf = np.array(X_hf)
        y_hf = np.array(y_hf)
        y_lf_at_hf = np.array(y_lf_at_hf)

        print(f"Training data prepared: {len(y_hf)} points")

        return X_hf, y_hf, y_lf_at_hf

    def fit(self, X_hf: np.ndarray, y_hf: np.ndarray, y_lf_at_hf: np.ndarray,
            training_iter: int = 50, use_batch_training: bool = False,
            batch_size: int = 10000):
        """
            Fit the multi-fidelity GP model on GPU with automatic memory management.

            Parameters
            ----------
            X_hf : np.ndarray
                High-fidelity input points (angle, x, y)
            y_hf : np.ndarray
                High-fidelity outputs (LES values)
            y_lf_at_hf : np.ndarray
                Low-fidelity outputs at HF points (RANS values)
            training_iter : int
                Number of training iterations for GP optimization
            use_batch_training : bool
                If True, use mini-batch training for very large datasets.
                Trains on random batches instead of full dataset.
            batch_size : int
                Size of mini-batches when use_batch_training=True
        """
        # If batch training requested and data is large, use batch method
        if use_batch_training and len(y_hf) > batch_size:
            return self.fit_batched(X_hf, y_hf, y_lf_at_hf, training_iter, batch_size)
        # Clear GPU memory before training
        self.clear_gpu_memory()
        self.print_memory_usage("Before training - ")

        # Check if subsampling is needed
        n_samples = len(y_hf)
        if n_samples > self.max_training_samples:
            print(f"\nWARNING: {n_samples:,} samples exceeds max {self.max_training_samples:,}")
            print(f"Randomly subsampling to prevent out-of-memory error...")

            indices = np.random.choice(n_samples, self.max_training_samples, replace=False)
            X_hf = X_hf[indices]
            y_hf = y_hf[indices]
            y_lf_at_hf = y_lf_at_hf[indices]

            print(f"Using {len(y_hf):,} samples for training")

        self.X_train_hf = X_hf
        self.y_train_hf = y_hf
        self.y_train_lf_at_hf = y_lf_at_hf

        # Filter out any NaN values that might have slipped through
        valid_mask = np.isfinite(y_hf) & np.isfinite(y_lf_at_hf) & (y_lf_at_hf > 1e-6)
        if valid_mask.sum() == 0:
            raise ValueError("No valid training data after filtering NaN values. Check your input data.")

        if valid_mask.sum() < len(y_hf):
            print(f"Filtered out {len(y_hf) - valid_mask.sum()} invalid/NaN training points")
            X_hf = X_hf[valid_mask]
            y_hf = y_hf[valid_mask]
            y_lf_at_hf = y_lf_at_hf[valid_mask]
            print(f"Using {len(y_hf):,} valid training points")

        self.X_train_hf = X_hf
        self.y_train_hf = y_hf
        self.y_train_lf_at_hf = y_lf_at_hf

        # Estimate scaling factor rho using least squares
        # y_hf ≈ rho * y_lf + delta
        self.rho = np.sum(y_hf * y_lf_at_hf) / np.sum(y_lf_at_hf**2)

        print(f"Estimated scaling factor rho: {self.rho:.4f}")

        # Compute residuals: delta = y_hf - rho * y_lf
        delta = y_hf - self.rho * y_lf_at_hf

        # Final check: ensure no NaN in residuals
        if not np.all(np.isfinite(delta)):
            raise ValueError("NaN values detected in residuals. Check your data for invalid values.")

        # Convert to torch tensors and move to device
        try:
            train_x = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
            train_y = torch.tensor(delta, dtype=torch.float32).to(self.device)
            self.print_memory_usage("After loading data - ")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nERROR: Out of memory when loading {len(X_hf):,} samples")
                print("Try reducing max_training_samples or use CPU mode")
                print("Example: model = MultiFidelityGPR_GPU(device='cpu')")
                raise
            else:
                raise

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        ).to(self.device)
        self.likelihood.noise = self.noise_level

        self.gp_model = ExactGPModel(
            train_x, train_y, self.likelihood, self.kernel_length_scale
        ).to(self.device)

        # Train the model
        print("Fitting GP on residuals (GPU-accelerated)...")
        self.gp_model.train()
        self.likelihood.train()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)

        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        # Training loop with progress indication and memory monitoring
        print(f"Training for {training_iter} iterations...")
        try:
            for i in range(training_iter):
                optimizer.zero_grad()
                output = self.gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")
                    if self.device.type == 'cuda' and (i + 1) % 20 == 0:
                        self.print_memory_usage("    ")

                optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nERROR: Out of memory during training iteration {i+1}")
                print("Try one of the following:")
                print("  1. Reduce training samples: model = MultiFidelityGPR_GPU(max_training_samples=10000)")
                print("  2. Use CPU mode: model = MultiFidelityGPR_GPU(device='cpu')")
                print("  3. Reduce training iterations: model.fit(..., training_iter=20)")
                self.clear_gpu_memory()
                raise
            else:
                raise

        # Set to eval mode
        self.gp_model.eval()
        self.likelihood.eval()

        # Store training data for later use
        self.X_train = train_x
        self.y_train = train_y

        print(f"GP fitted successfully on {self.device}")
        print(f"Final noise level: {self.likelihood.noise.item():.6f}")
        print(f"Lengthscales: {self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
        self.print_memory_usage("After training - ")

        # Clear unnecessary tensors
        del train_x, train_y
        self.clear_gpu_memory()

    def fit_batched(self, X_hf: np.ndarray, y_hf: np.ndarray, y_lf_at_hf: np.ndarray,
                    training_iter: int = 50, batch_size: int = 10000):
        """
            Fit the multi-fidelity GP model using mini-batch training.

            This method is useful for very large datasets that don't fit in GPU memory.
            It trains on random batches of data, which uses less memory but may be
            slightly less accurate than training on all data at once.

            Parameters
            ----------
            X_hf : np.ndarray
                High-fidelity input points (angle, x, y)
            y_hf : np.ndarray
                High-fidelity outputs (LES values)
            y_lf_at_hf : np.ndarray
                Low-fidelity outputs at HF points (RANS values)
            training_iter : int
                Number of training iterations (each iteration uses one batch)
            batch_size : int
                Number of samples per batch
        """
        print("="*70)
        print("BATCH TRAINING MODE")
        print("="*70)

        self.clear_gpu_memory()
        self.print_memory_usage("Before batch training - ")

        n_samples = len(y_hf)
        print(f"\nTotal samples: {n_samples:,}")
        print(f"Batch size: {batch_size:,}")
        print(f"Memory-efficient training on random batches")

        # Filter out any NaN values in batch training mode
        valid_mask = np.isfinite(y_hf) & np.isfinite(y_lf_at_hf) & (y_lf_at_hf > 1e-6)
        if valid_mask.sum() == 0:
            raise ValueError("No valid training data after filtering NaN values. Check your input data.")

        if valid_mask.sum() < len(y_hf):
            print(f"Filtered out {len(y_hf) - valid_mask.sum()} invalid/NaN training points")
            X_hf = X_hf[valid_mask]
            y_hf = y_hf[valid_mask]
            y_lf_at_hf = y_lf_at_hf[valid_mask]
            n_samples = len(y_hf)
            print(f"Using {n_samples:,} valid training points")

        self.X_train_hf = X_hf
        self.y_train_hf = y_hf
        self.y_train_lf_at_hf = y_lf_at_hf

        # Estimate scaling factor rho using all data (lightweight operation)
        print("\nEstimating scaling factor rho...")
        self.rho = np.sum(y_hf * y_lf_at_hf) / np.sum(y_lf_at_hf**2)
        print(f"Estimated scaling factor rho: {self.rho:.4f}")

        # Compute residuals for all data
        delta = y_hf - self.rho * y_lf_at_hf

        # Final check: ensure no NaN in residuals
        if not np.all(np.isfinite(delta)):
            raise ValueError("NaN values detected in residuals. Check your data for invalid values.")

        # Initialize with first batch
        print(f"\nInitializing model with first batch ({batch_size:,} samples)...")
        indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        train_x_init = torch.tensor(X_hf[indices], dtype=torch.float32).to(self.device)
        train_y_init = torch.tensor(delta[indices], dtype=torch.float32).to(self.device)

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        ).to(self.device)
        self.likelihood.noise = self.noise_level

        self.gp_model = ExactGPModel(
            train_x_init, train_y_init, self.likelihood, self.kernel_length_scale
        ).to(self.device)

        self.print_memory_usage("After initialization - ")

        # Train using mini-batches
        print(f"\nTraining on random batches for {training_iter} iterations...")
        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        losses = []

        try:
            for i in range(training_iter):
                # Sample a random batch
                batch_indices = np.random.choice(n_samples, batch_size, replace=False)
                batch_x = torch.tensor(X_hf[batch_indices], dtype=torch.float32).to(self.device)
                batch_y = torch.tensor(delta[batch_indices], dtype=torch.float32).to(self.device)

                # Update model's training data for this batch
                self.gp_model.set_train_data(batch_x, batch_y, strict=False)

                # Training step
                optimizer.zero_grad()
                output = self.gp_model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # Progress reporting
                if (i + 1) % 10 == 0 or i == 0:
                    avg_loss = np.mean(losses[-10:])
                    print(f"  Iter {i+1}/{training_iter} - Loss: {loss.item():.3f} (avg: {avg_loss:.3f})")
                    if self.device.type == 'cuda' and (i + 1) % 20 == 0:
                        self.print_memory_usage("    ")

                # Clear batch tensors to free memory
                del batch_x, batch_y
                if (i + 1) % 10 == 0:
                    self.clear_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nERROR: Out of memory during batch training iteration {i+1}")
                print("Try reducing batch_size:")
                print(f"  model.fit(..., use_batch_training=True, batch_size={batch_size//2})")
                self.clear_gpu_memory()
                raise
            else:
                raise

        # Use final batch for model storage
        final_indices = np.random.choice(n_samples, batch_size, replace=False)
        self.X_train = torch.tensor(X_hf[final_indices], dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(delta[final_indices], dtype=torch.float32).to(self.device)
        self.gp_model.set_train_data(self.X_train, self.y_train, strict=False)

        # Set to eval mode
        self.gp_model.eval()
        self.likelihood.eval()

        print(f"\nBatch training complete!")
        print(f"Final loss: {losses[-1]:.3f} (avg last 10: {np.mean(losses[-10:]):.3f})")
        print(f"Final noise level: {self.likelihood.noise.item():.6f}")
        print(f"Lengthscales: {self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
        self.print_memory_usage("After batch training - ")

        del train_x_init, train_y_init
        self.clear_gpu_memory()

        print("="*70)

    def predict(self, X_test: np.ndarray, data: Dict,
                return_std: bool = True, batch_size: int = 10000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
            Predict at test points using the multi-fidelity model (GPU-accelerated).

            Parameters
            ----------
            X_test : np.ndarray
                Test points (angle, x, y)
            data : dict
                Dictionary containing RANS data for low-fidelity predictions
            return_std : bool
                Whether to return standard deviation
            batch_size : int
                Batch size for GPU processing (to avoid memory issues)

            Returns
            -------
            y_pred : np.ndarray
                Predicted values
            std_pred : np.ndarray, optional
                Prediction standard deviations
        """
        # Get low-fidelity predictions at test points
        y_lf_test = self._get_rans_at_points(X_test, data)

        # Process in batches to avoid GPU memory issues
        n_test = len(X_test)
        n_batches = int(np.ceil(n_test / batch_size))

        delta_pred_list = []
        delta_std_list = [] if return_std else None

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_test)

                X_batch = torch.tensor(
                    X_test[start_idx:end_idx], dtype=torch.float32
                ).to(self.device)

                # GP prediction
                pred = self.likelihood(self.gp_model(X_batch))

                delta_pred_list.append(pred.mean.cpu().numpy())
                if return_std:
                    delta_std_list.append(pred.stddev.cpu().numpy())

        # Concatenate batches
        delta_pred = np.concatenate(delta_pred_list)
        delta_std = np.concatenate(delta_std_list) if return_std else None

        # Multi-fidelity prediction: y = rho * y_lf + delta
        y_pred = self.rho * y_lf_test + delta_pred

        # Handle NaN values from RANS interpolation (building regions)
        # Set predictions to 0 where RANS is NaN (these are building/obstacle regions)
        nan_mask = ~np.isfinite(y_lf_test)
        if nan_mask.any():
            y_pred[nan_mask] = 0.0
            if return_std:
                delta_std[nan_mask] = 0.0

        if return_std:
            std_pred = delta_std
            return y_pred, std_pred
        else:
            return y_pred, None

    def _get_rans_at_points(self, X: np.ndarray, data: Dict) -> np.ndarray:
        """
            Get RANS values at arbitrary points via interpolation.

            Parameters
            ----------
            X : np.ndarray
                Points (angle, x, y)
            data : dict
                Dictionary containing RANS data

            Returns
            -------
            rans_values : np.ndarray
                Interpolated RANS values at X
        """
        rans_data = data['rans']['data']
        x_grid = data['grid']['x']
        y_grid = data['grid']['y']

        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

        rans_values = np.zeros(len(X))

        for i, (angle, x, y) in enumerate(X):
            # Interpolate in angle
            angle_idx_low = int(np.floor(angle)) % 360
            angle_idx_high = int(np.ceil(angle)) % 360
            alpha = angle - np.floor(angle)

            rans_field = (1 - alpha) * rans_data[angle_idx_low] + \
                         alpha * rans_data[angle_idx_high]

            # Interpolate in space
            points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
            values = rans_field.ravel()

            # Mask out NaN/invalid values before interpolation to prevent NaN leakage
            valid_mask = np.isfinite(values)
            if valid_mask.sum() > 0:
                rans_values[i] = griddata(points[valid_mask], values[valid_mask],
                                         np.array([[x, y]]), method='linear')[0]
            else:
                rans_values[i] = np.nan

        return rans_values

    def predict_field(self, angle: float, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
            Predict wind field at a specific angle.

            Parameters
            ----------
            angle : float
                Wind angle in degrees
            data : dict
                Dictionary containing grid and RANS data

            Returns
            -------
            prediction : np.ndarray
                Predicted wind field (nx, ny)
            uncertainty : np.ndarray
                Prediction uncertainty (nx, ny)
        """
        x_grid = data['grid']['x']
        y_grid = data['grid']['y']
        nx, ny = len(x_grid), len(y_grid)

        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

        # Create test points
        X_test = np.column_stack([
            np.full(nx * ny, angle),
            X_mesh.ravel(),
            Y_mesh.ravel()
        ])

        # Get RANS field directly (much faster than point-by-point interpolation)
        rans_data = data['rans']['data']
        angle_idx_low = int(np.floor(angle)) % 360
        angle_idx_high = int(np.ceil(angle)) % 360
        alpha = angle - np.floor(angle)
        rans_field = (1 - alpha) * rans_data[angle_idx_low] + alpha * rans_data[angle_idx_high]
        y_lf_test = rans_field.ravel()

        # Process in batches to avoid GPU memory issues
        n_test = len(X_test)
        batch_size = 10000
        n_batches = int(np.ceil(n_test / batch_size))

        delta_pred_list = []
        delta_std_list = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_test)

                X_batch = torch.tensor(
                    X_test[start_idx:end_idx], dtype=torch.float32
                ).to(self.device)

                # GP prediction
                pred = self.likelihood(self.gp_model(X_batch))

                delta_pred_list.append(pred.mean.cpu().numpy())
                delta_std_list.append(pred.stddev.cpu().numpy())

        # Concatenate batches
        delta_pred = np.concatenate(delta_pred_list)
        delta_std = np.concatenate(delta_std_list)

        # Multi-fidelity prediction: y = rho * y_lf + delta
        y_pred = self.rho * y_lf_test + delta_pred

        # Handle NaN values from RANS interpolation (building regions)
        nan_mask = ~np.isfinite(y_lf_test)
        if nan_mask.any():
            y_pred[nan_mask] = 0.0
            delta_std[nan_mask] = 0.0

        # Reshape to grid
        prediction = y_pred.reshape(nx, ny)
        uncertainty = delta_std.reshape(nx, ny)

        # Replace any remaining NaN values with 0
        prediction = np.nan_to_num(prediction, nan=0.0)
        uncertainty = np.nan_to_num(uncertainty, nan=0.0)

        return prediction, uncertainty

    def save_model(self, filename: str):
        """
            Save trained model to file.

            Parameters
            ----------
            filename : str
                Path to save the model (e.g., 'model.pth' or 'model.pkl')
        """
        if self.gp_model is None or self.rho is None:
            raise ValueError("Model must be trained before saving. Call fit() first.")

        # Prepare model state
        model_state = {
            'gp_model_state': self.gp_model.state_dict(),
            'likelihood_state': self.likelihood.state_dict(),
            'rho': self.rho,
            'kernel_length_scale': self.kernel_length_scale,
            'noise_level': self.noise_level,
            'device': str(self.device),
            'X_train': self.X_train.cpu().numpy() if self.X_train is not None else None,
            'y_train': self.y_train.cpu().numpy() if self.y_train is not None else None,
        }

        # Use torch.save for better compatibility with PyTorch models
        torch.save(model_state, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str, X_train_shape: Optional[int] = None):
        """
            Load trained model from file.

            Parameters
            ----------
            filename : str
                Path to the saved model file
            X_train_shape : int, optional
                Number of training points (needed for reconstruction).
                If None, will try to load from saved state.

            Returns
            -------
            self : MultiFidelityGPR_GPU
                Returns self for method chaining
        """
        model_state = torch.load(filename, map_location=self.device, weights_only=False)

        # Restore hyperparameters
        self.rho = model_state['rho']
        self.kernel_length_scale = model_state['kernel_length_scale']
        self.noise_level = model_state['noise_level']

        # Restore training data if available
        if model_state['X_train'] is not None and model_state['y_train'] is not None:
            self.X_train = torch.tensor(model_state['X_train'], dtype=torch.float32).to(self.device)
            self.y_train = torch.tensor(model_state['y_train'], dtype=torch.float32).to(self.device)

            # Recreate likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp_model = ExactGPModel(
                self.X_train, self.y_train, self.likelihood, self.kernel_length_scale
            ).to(self.device)

            # Load state dictionaries
            self.gp_model.load_state_dict(model_state['gp_model_state'])
            self.likelihood.load_state_dict(model_state['likelihood_state'])

            # Set to evaluation mode
            self.gp_model.eval()
            self.likelihood.eval()

            print(f"Model loaded from {filename}")
        else:
            print(f"Warning: Training data not found in {filename}.")
            print("You'll need to retrain or provide training data to use predictions.")

        return self


def main():
    """
    Main function demonstrating the GPU-accelerated multi-fidelity GPR workflow.
    """
    print("=" * 70)
    print("GPU-Accelerated Multi-Fidelity Gaussian Process Regression")
    print("=" * 70)

    # Initialize model (auto-detects GPU)
    mf_gpr = MultiFidelityGPR_GPU(
        kernel_length_scale=15.0,
        noise_level=0.01,
        device='auto'  # Use 'cuda' to force GPU, 'cpu' to force CPU
    )

    # Load data
    print("\n1. Loading data...")
    x_grid, y_grid, data = mf_gpr.load_data()

    # Prepare training data
    print("\n2. Preparing training data...")
    X_hf, y_hf, y_lf_at_hf = mf_gpr.prepare_training_data(data)

    # Option to subsample for faster training
    print("\n3. Training model...")
    print("   Note: GPU acceleration allows training on more data points!")

    # For GPU, we can use more data (e.g., 20% instead of 10%)
    n_samples = len(y_hf)
    subsample_ratio = 0.2  # Use 20% of data
    n_subsample = int(n_samples * subsample_ratio)

    print(f"   Using {n_subsample} of {n_samples} points ({subsample_ratio*100}%)")
    indices = np.random.choice(n_samples, n_subsample, replace=False)
    X_hf_sub = X_hf[indices]
    y_hf_sub = y_hf[indices]
    y_lf_at_hf_sub = y_lf_at_hf[indices]

    # Fit model
    import time
    start_time = time.time()
    mf_gpr.fit(X_hf_sub, y_hf_sub, y_lf_at_hf_sub, training_iter=50)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Test predictions at some angles
    print("\n4. Making predictions...")
    test_angles = [10.0, 45.0, 90.0, 180.0, 270.0]

    fig, axes = plt.subplots(len(test_angles), 3, figsize=(15, 4*len(test_angles)))

    for i, angle in enumerate(test_angles):
        print(f"   Predicting at angle {angle}°...")

        # Get prediction
        pred, std = mf_gpr.predict_field(angle, data)

        # Get RANS for comparison
        angle_idx = int(angle) % 360
        rans_field = data['rans']['data'][angle_idx]

        # Plot
        vmin, vmax = 0, 0.8

        axes[i, 0].contourf(x_grid, y_grid, rans_field.T, levels=20,
                           vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 0].set_title(f'RANS (Low-Fidelity) - {angle}°')
        axes[i, 0].set_aspect('equal')

        axes[i, 1].contourf(x_grid, y_grid, pred.T, levels=20,
                           vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 1].set_title(f'MF-GPR Prediction (GPU) - {angle}°')
        axes[i, 1].set_aspect('equal')

        im = axes[i, 2].contourf(x_grid, y_grid, std.T, levels=20, cmap='Reds')
        axes[i, 2].set_title(f'Prediction Uncertainty - {angle}°')
        axes[i, 2].set_aspect('equal')
        plt.colorbar(im, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig('multifidelity_predictions_gpu.png', dpi=150, bbox_inches='tight')
    print("\n5. Results saved to 'multifidelity_predictions_gpu.png'")

    # Validation: compare with actual LES data
    print("\n6. Validation against LES data...")
    les_angles = data['les']['angles']
    les_data = data['les']['data']

    errors = []
    for i, angle in enumerate(les_angles[:5]):  # Test on first 5 angles
        pred, _ = mf_gpr.predict_field(angle, data)
        les_field = les_data[i]

        # Compute RMSE on non-zero regions
        mask = les_field > 1e-6
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((pred[mask] - les_field[mask])**2))
            errors.append(rmse)
            print(f"   Angle {angle:6.2f}°: RMSE = {rmse:.6f}")

    print(f"\n   Mean RMSE: {np.mean(errors):.6f}")
    print("\n" + "=" * 70)
    print("GPU-accelerated multi-fidelity GPR analysis complete!")
    print("=" * 70)

    return mf_gpr, data


if __name__ == "__main__":
    model, data = main()

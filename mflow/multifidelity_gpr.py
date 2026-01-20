import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import warnings
import pickle
warnings.filterwarnings('ignore')


class MultiFidelityGPR:
    """
        Multi-Fidelity Gaussian Process Regression for wind field prediction.

        Implements the co-kriging approach:
        y_high(x) = rho * y_low(x) + delta(x)

        where:
        - y_high: High-fidelity (LES) data
        - y_low: Low-fidelity (RANS) data
        - rho: Scaling coefficient
        - delta: GP-based correction term
    """

    def __init__(self, kernel_length_scale: float = 10.0, noise_level: float = 0.01):
        """
            Initialize the multi-fidelity GPR model.

            Parameters
            ----------
            kernel_length_scale : float
                Length scale for RBF kernel (controls smoothness)
            noise_level : float
                White noise level for numerical stability
        """
        self.kernel_length_scale = kernel_length_scale
        self.noise_level = noise_level

        # Kernel: combines RBF for smoothness + white noise for stability
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=[kernel_length_scale, kernel_length_scale, kernel_length_scale],
            length_scale_bounds=(1.0, 100.0)
        ) + WhiteKernel(noise_level=noise_level)

        self.gp_delta = None
        self.rho = None
        self.X_train_hf = None
        self.y_train_hf = None
        self.y_train_lf_at_hf = None
    #    
    # This function needs to be updated when non-standard data format/layout is used    
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
        # Load spatial grid
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
        rans_angles = np.arange(1, 361, 1)  # 1 to 359 degrees
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

    def fit(self, X_hf: np.ndarray, y_hf: np.ndarray, y_lf_at_hf: np.ndarray):
        """
            Fit the multi-fidelity GP model.

            Parameters
            ----------
            X_hf : np.ndarray
                High-fidelity input points (angle, x, y)
            y_hf : np.ndarray
                High-fidelity outputs (LES values)
            y_lf_at_hf : np.ndarray
                Low-fidelity outputs at HF points (RANS values)
        """
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
        # y_hf = rho * y_lf + delta
        self.rho = np.sum(y_hf * y_lf_at_hf) / np.sum(y_lf_at_hf**2)

        print(f"Estimated scaling factor rho: {self.rho:.4f}")

        # Compute residuals: delta = y_hf - rho * y_lf
        delta = y_hf - self.rho * y_lf_at_hf

        # Final check: ensure no NaN in residuals
        if not np.all(np.isfinite(delta)):
            raise ValueError("NaN values detected in residuals. Check your data for invalid values.")

        # Fit GP on the residuals
        print("Fitting GP on residuals...")
        self.gp_delta = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            alpha=1e-6
        )
        self.gp_delta.fit(X_hf, delta)

        print(f"GP fitted. Kernel: {self.gp_delta.kernel_}")
        print(f"Log-likelihood: {self.gp_delta.log_marginal_likelihood_value_:.2f}")

    def predict(self, X_test: np.ndarray, data: Dict,
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
            Predict at test points using the multi-fidelity model.

            Parameters
            ----------
            X_test : np.ndarray
                Test points (angle, x, y)
            data : dict
                Dictionary containing RANS data for low-fidelity predictions
            return_std : bool
                Whether to return standard deviation

            Returns
            -------
            y_pred : np.ndarray
                Predicted values
            std_pred : np.ndarray, optional
                Prediction standard deviations
        """
        # Get low-fidelity predictions at test points
        y_lf_test = self._get_rans_at_points(X_test, data)

        # Get GP prediction of residuals
        if return_std:
            delta_pred, delta_std = self.gp_delta.predict(X_test, return_std=True)
        else:
            delta_pred = self.gp_delta.predict(X_test, return_std=False)
            delta_std = None

        # Multi-fidelity prediction: y = rho * y_lf + delta
        y_pred = self.rho * y_lf_test + delta_pred

        # Handle NaN values from RANS interpolation (building regions)
        # Set predictions to 0 where RANS is NaN (these are building/obstacle regions)
        nan_mask = ~np.isfinite(y_lf_test)
        if nan_mask.any():
            y_pred[nan_mask] = 0.0
            if return_std and delta_std is not None:
                delta_std[nan_mask] = 0.0

        if return_std:
            # Uncertainty propagation (simplified)
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

        # Predict
        y_pred, std_pred = self.predict(X_test, data, return_std=True)

        # Reshape to grid
        prediction = y_pred.reshape(nx, ny)
        uncertainty = std_pred.reshape(nx, ny)

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
                Path to save the model (e.g., 'model.pkl')
        """
        if self.gp_delta is None or self.rho is None:
            raise ValueError("Model must be trained before saving. Call fit() first.")

        model_state = {
            'gp_delta': self.gp_delta,
            'rho': self.rho,
            'kernel_length_scale': self.kernel_length_scale,
            'noise_level': self.noise_level,
            'subsample_ratio': self.subsample_ratio,
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """
            Load trained model from file.

            Parameters
            ----------
            filename : str
                Path to the saved model file

            Returns
            -------
            self : MultiFidelityGPR
                Returns self for method chaining
        """
        with open(filename, 'rb') as f:
            model_state = pickle.load(f)

        self.gp_delta = model_state['gp_delta']
        self.rho = model_state['rho']
        self.kernel_length_scale = model_state['kernel_length_scale']
        self.noise_level = model_state['noise_level']
        self.subsample_ratio = model_state['subsample_ratio']

        print(f"Model loaded from {filename}")
        return self


def main():
    """
    Main function demonstrating the multi-fidelity GPR workflow.
    """
    print("=" * 70)
    print("Multi-Fidelity Gaussian Process Regression for Wind Prediction")
    print("=" * 70)

    # Initialize model
    mf_gpr = MultiFidelityGPR(kernel_length_scale=15.0, noise_level=0.01)

    # Load data
    print("\n1. Loading data...")
    x_grid, y_grid, data = mf_gpr.load_data()

    # Prepare training data
    print("\n2. Preparing training data...")
    X_hf, y_hf, y_lf_at_hf = mf_gpr.prepare_training_data(data)

    # Option to subsample for faster training
    print("\n3. Training model...")
    print("   Note: Training on all points may take several minutes.")
    print("   Consider subsampling for faster initial testing.")

    # Subsample to 10% for demonstration (remove for full accuracy)
    n_samples = len(y_hf)
    subsample_ratio = 0.1  # Use 10% of data
    n_subsample = int(n_samples * subsample_ratio)

    if n_subsample < n_samples:
        print(f"   Subsampling: using {n_subsample} of {n_samples} points ({subsample_ratio*100}%)")
        indices = np.random.choice(n_samples, n_subsample, replace=False)
        X_hf_sub = X_hf[indices]
        y_hf_sub = y_hf[indices]
        y_lf_at_hf_sub = y_lf_at_hf[indices]
    else:
        X_hf_sub = X_hf
        y_hf_sub = y_hf
        y_lf_at_hf_sub = y_lf_at_hf

    # Fit model
    mf_gpr.fit(X_hf_sub, y_hf_sub, y_lf_at_hf_sub)

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
        axes[i, 1].set_title(f'MF-GPR Prediction - {angle}°')
        axes[i, 1].set_aspect('equal')

        im = axes[i, 2].contourf(x_grid, y_grid, std.T, levels=20, cmap='Reds')
        axes[i, 2].set_title(f'Prediction Uncertainty - {angle}°')
        axes[i, 2].set_aspect('equal')
        plt.colorbar(im, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig('multifidelity_predictions.png', dpi=150, bbox_inches='tight')
    print("\n5. Results saved to 'multifidelity_predictions.png'")

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
    print("Multi-fidelity GPR analysis complete!")
    print("=" * 70)

    return mf_gpr, data


if __name__ == "__main__":
    model, data = main()

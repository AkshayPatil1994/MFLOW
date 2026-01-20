import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import pickle
#
# Advanced Multi-Fidelity Gaussian Process Regression (GPR)
#
class AdvancedMultiFidelityGPR:
    """
    Advanced Multi-Fidelity GPR with optimized kernels and efficient computation.
    """

    def __init__(self, spatial_length_scale: float = 50.0,
                 angular_length_scale: float = 10.0,
                 use_matern: bool = True):
        """
            Initialize advanced multi-fidelity GPR.

            Parameters
            ----------
            spatial_length_scale : float
                Initial length scale for spatial dimensions (x, y)
            angular_length_scale : float
                Initial length scale for angular dimension
            use_matern : bool
                Use Matern kernel (more flexible) vs RBF
        """
        self.spatial_length_scale = spatial_length_scale
        self.angular_length_scale = angular_length_scale
        self.use_matern = use_matern

        # Build anisotropic kernel with different length scales for angle vs space
        if use_matern:
            kernel_base = Matern(
                length_scale=[angular_length_scale, spatial_length_scale, spatial_length_scale],
                length_scale_bounds=[(1.0, 50.0), (10.0, 200.0), (10.0, 200.0)],
                nu=2.5
            )
        else:
            kernel_base = RBF(
                length_scale=[angular_length_scale, spatial_length_scale, spatial_length_scale],
                length_scale_bounds=[(1.0, 50.0), (10.0, 200.0), (10.0, 200.0)]
            )

        self.kernel = C(1.0, (1e-2, 1e2)) * kernel_base + WhiteKernel(
            noise_level=0.01, noise_level_bounds=(1e-5, 1e-1)
        )

        self.gp_delta = None
        self.rho_field = None  # Spatial and angular varying rho
        self.data = None

    def load_data(self, data_dir: str = 'truncated_data') -> Dict:
        """Load and organize all data."""
        # Load spatial grid
        x_grid = np.loadtxt(f'{data_dir}/x_common_grid.txt')
        y_grid = np.loadtxt(f'{data_dir}/y_common_grid.txt')

        # LES collocation points
        les_angles = np.array([0.59, 4.37, 16.57, 37.28, 51.62, 57.55, 61.7, 65.28,
                               69.16, 73.95, 80.24, 88.15, 97.2, 105.51, 110.89, 114.42,
                               117.54, 121.19, 126.45, 133.31, 139.9, 149.34, 163.13, 172.08,
                               177.08, 180.76, 185.65, 192.73, 203.04, 216.59, 237.0, 261.88,
                               275.8, 283.3, 290.15, 298.0, 310.01, 325.46, 336.53, 341.85,
                               345.11, 347.46, 349.42, 351.3, 353.16, 355.14, 357.14, 358.86])

        exp_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

        # Load LES data
        les_data = []
        for exp_num in exp_numbers:
            filename = f'{data_dir}/U_les_normalised_{exp_num:03d}.txt'
            data_field = np.loadtxt(filename, skiprows=1)
            les_data.append(data_field)
        les_data = np.array(les_data)

        # Load RANS data
        rans_angles = np.arange(1, 361, 1)  # 1 to 360 degrees
        rans_data = []
        for angle in rans_angles:
            filename = f'{data_dir}/U_rans_normalised_{angle:03d}.txt'
            data_field = np.loadtxt(filename, skiprows=1)
            rans_data.append(data_field)
        rans_data = np.array(rans_data)

        self.data = {
            'les': {'angles': les_angles, 'data': les_data},
            'rans': {'angles': rans_angles, 'data': rans_data},
            'grid': {'x': x_grid, 'y': y_grid}
        }

        print(f"Data loaded:")
        print(f"  LES: {len(les_angles)} angles, {les_data.shape[1]}×{les_data.shape[2]} grid")
        print(f"  RANS: {len(rans_angles)} angles, {rans_data.shape[1]}×{rans_data.shape[2]} grid")
        print(f"  Spatial domain: x ∈ [{x_grid.min():.1f}, {x_grid.max():.1f}], "
              f"y ∈ [{y_grid.min():.1f}, {y_grid.max():.1f}]")

        return self.data

    def prepare_training_data(self, subsample_spatial: int = 1) -> Tuple:
        """
            Prepare training data with optional spatial subsampling.

            Parameters
            ----------
            subsample_spatial : int
                Subsample every Nth spatial point (1 = all points)

            Returns
            -------
            X_hf, y_hf, y_lf : Training data arrays
        """
        les_angles = self.data['les']['angles']
        les_data = self.data['les']['data']
        rans_data = self.data['rans']['data']
        x_grid = self.data['grid']['x']
        y_grid = self.data['grid']['y']

        # Subsample spatial grid
        x_sub = x_grid[::subsample_spatial]
        y_sub = y_grid[::subsample_spatial]
        nx, ny = len(x_sub), len(y_sub)

        X_mesh, Y_mesh = np.meshgrid(x_sub, y_sub, indexing='ij')

        X_hf, y_hf, y_lf = [], [], []

        for i, angle in enumerate(les_angles):
            les_field = les_data[i][::subsample_spatial, ::subsample_spatial]

            # Interpolate RANS to this angle
            angle_low = int(np.floor(angle)) % 360
            angle_high = int(np.ceil(angle)) % 360
            alpha = angle - np.floor(angle)

            rans_field = (1 - alpha) * rans_data[angle_low][::subsample_spatial, ::subsample_spatial] + \
                         alpha * rans_data[angle_high][::subsample_spatial, ::subsample_spatial]

            for ix in range(nx):
                for iy in range(ny):
                    if les_field[ix, iy] > 1e-6 and rans_field[ix, iy] > 1e-6:
                        X_hf.append([angle, X_mesh[ix, iy], Y_mesh[ix, iy]])
                        y_hf.append(les_field[ix, iy])
                        y_lf.append(rans_field[ix, iy])

        X_hf = np.array(X_hf)
        y_hf = np.array(y_hf)
        y_lf = np.array(y_lf)

        print(f"Training data: {len(y_hf)} points (spatial subsample: {subsample_spatial})")

        return X_hf, y_hf, y_lf

    def fit(self, X_hf: np.ndarray, y_hf: np.ndarray, y_lf: np.ndarray,
            method: str = 'autoregressive'):
        """
            Fit multi-fidelity model.

            Parameters
            ----------
            X_hf : np.ndarray
                High-fidelity inputs (angle, x, y)
            y_hf : np.ndarray
                High-fidelity outputs (LES)
            y_lf : np.ndarray
                Low-fidelity outputs (RANS)
            method : str
                'autoregressive' or 'global_scaling'
        """
        if method == 'global_scaling':
            # Simple global scaling factor
            self.rho = np.sum(y_hf * y_lf) / np.sum(y_lf**2)
            delta = y_hf - self.rho * y_lf
            print(f"Global scaling factor ρ = {self.rho:.4f}")

        elif method == 'autoregressive':
            # Autoregressive: delta = y_hf - rho(x) * y_lf where rho varies
            # Approximate local rho
            from sklearn.linear_model import Ridge

            # Reshape for regression
            y_lf_reshaped = y_lf.reshape(-1, 1)

            # Fit local scaling
            ridge = Ridge(alpha=1.0)
            ridge.fit(y_lf_reshaped, y_hf)

            self.rho = ridge.coef_[0]
            delta = y_hf - ridge.predict(y_lf_reshaped)
            print(f"Autoregressive scaling factor ρ ≈ {self.rho:.4f}")

        # Fit GP on residuals
        print("Fitting GP on residuals...")
        self.gp_delta = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-6
        )

        self.gp_delta.fit(X_hf, delta)
        print(f"Optimized kernel: {self.gp_delta.kernel_}")
        print(f"Log marginal likelihood: {self.gp_delta.log_marginal_likelihood_value_:.2f}")

        # Store training data for later use
        self.X_train = X_hf
        self.y_train_lf = y_lf

    def predict(self, X_test: np.ndarray, return_std: bool = True) -> Tuple:
        """Predict at test points."""
        # Get RANS at test points
        y_lf_test = self._interpolate_rans(X_test)

        # Get GP prediction
        if return_std:
            delta_pred, delta_std = self.gp_delta.predict(X_test, return_std=True)
        else:
            delta_pred = self.gp_delta.predict(X_test, return_std=False)
            delta_std = None

        # Combine: y = rho * y_lf + delta
        y_pred = self.rho * y_lf_test + delta_pred

        return y_pred, delta_std

    def _interpolate_rans(self, X: np.ndarray) -> np.ndarray:
        """Interpolate RANS data at arbitrary points."""
        from scipy.interpolate import RegularGridInterpolator

        rans_data = self.data['rans']['data']
        x_grid = self.data['grid']['x']
        y_grid = self.data['grid']['y']
        rans_angles = self.data['rans']['angles']

        # Create 3D interpolator (angle, x, y)
        interpolator = RegularGridInterpolator(
            (rans_angles, x_grid, y_grid),
            rans_data,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Handle angle periodicity
        X_periodic = X.copy()
        X_periodic[:, 0] = X_periodic[:, 0] % 360

        return interpolator(X_periodic)

    def predict_field(self, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict full field at given angle."""
        x_grid = self.data['grid']['x']
        y_grid = self.data['grid']['y']
        nx, ny = len(x_grid), len(y_grid)

        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
        X_test = np.column_stack([
            np.full(nx * ny, angle),
            X_mesh.ravel(),
            Y_mesh.ravel()
        ])

        y_pred, std_pred = self.predict(X_test, return_std=True)

        return y_pred.reshape(nx, ny), std_pred.reshape(nx, ny)

    def cross_validate(self, n_folds: int = 5) -> Dict:
        """
            Perform k-fold cross-validation.

            Returns
            -------
            results : dict
                CV scores and metrics
        """
        from sklearn.model_selection import KFold

        X_hf, y_hf, y_lf = self.prepare_training_data(subsample_spatial=2)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = {'rmse': [], 'mae': [], 'r2': []}

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_hf)):
            print(f"  Fold {fold + 1}/{n_folds}...", end=' ')

            X_train, X_test = X_hf[train_idx], X_hf[test_idx]
            y_train, y_test = y_hf[train_idx], y_hf[test_idx]
            y_lf_train, y_lf_test = y_lf[train_idx], y_lf[test_idx]

            self.fit(X_train, y_train, y_lf_train)
            y_pred, _ = self.predict(X_test, return_std=False)

            rmse = np.sqrt(np.mean((y_pred - y_test)**2))
            mae = np.mean(np.abs(y_pred - y_test))
            r2 = 1 - np.sum((y_pred - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)

            scores['rmse'].append(rmse)
            scores['mae'].append(mae)
            scores['r2'].append(r2)

            print(f"RMSE={rmse:.5f}, MAE={mae:.5f}, R²={r2:.4f}")

        print(f"\nCross-validation results:")
        print(f"  RMSE: {np.mean(scores['rmse']):.5f} ± {np.std(scores['rmse']):.5f}")
        print(f"  MAE:  {np.mean(scores['mae']):.5f} ± {np.std(scores['mae']):.5f}")
        print(f"  R²:   {np.mean(scores['r2']):.4f} ± {np.std(scores['r2']):.4f}")

        return scores

    def save_model(self, filename: str):
        """Save trained model to file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'gp_delta': self.gp_delta,
                'rho': self.rho,
                'kernel': self.kernel,
                'data_summary': {
                    'les_angles': self.data['les']['angles'],
                    'grid_shape': self.data['les']['data'].shape
                }
            }, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """Load trained model from file."""
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
            self.gp_delta = saved['gp_delta']
            self.rho = saved['rho']
            self.kernel = saved['kernel']
        print(f"Model loaded from {filename}")


def create_diagnostic_plots(model: AdvancedMultiFidelityGPR, test_angles: List[float]):
    """Create comprehensive diagnostic plots."""
    fig = plt.figure(figsize=(18, 12))

    n_angles = len(test_angles)
    for i, angle in enumerate(test_angles):
        # Predictions
        pred, std = model.predict_field(angle)

        # RANS comparison
        angle_idx = int(angle) % 360
        rans = model.data['rans']['data'][angle_idx]

        # Find closest LES if available
        les_angles = model.data['les']['angles']
        if np.min(np.abs(les_angles - angle)) < 0.5:
            les_idx = np.argmin(np.abs(les_angles - angle))
            les = model.data['les']['data'][les_idx]
            has_les = True
        else:
            has_les = False

        x_grid = model.data['grid']['x']
        y_grid = model.data['grid']['y']

        # RANS
        ax1 = plt.subplot(n_angles, 4, i*4 + 1)
        im1 = ax1.contourf(x_grid, y_grid, rans.T, levels=20, cmap='viridis')
        ax1.set_title(f'RANS - {angle}°')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        # MF-GPR
        ax2 = plt.subplot(n_angles, 4, i*4 + 2)
        im2 = ax2.contourf(x_grid, y_grid, pred.T, levels=20, cmap='viridis')
        ax2.set_title(f'MF-GPR - {angle}°')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

        # Uncertainty
        ax3 = plt.subplot(n_angles, 4, i*4 + 3)
        im3 = ax3.contourf(x_grid, y_grid, std.T, levels=20, cmap='Reds')
        ax3.set_title(f'Uncertainty - {angle}°')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)

        # Error if LES available
        ax4 = plt.subplot(n_angles, 4, i*4 + 4)
        if has_les:
            error = np.abs(pred - les)
            im4 = ax4.contourf(x_grid, y_grid, error.T, levels=20, cmap='Reds')
            ax4.set_title(f'|Error| vs LES - {angle}°')
            rmse = np.sqrt(np.mean(error[les > 1e-6]**2))
            ax4.text(0.5, 0.95, f'RMSE={rmse:.4f}', transform=ax4.transAxes,
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No LES data', transform=ax4.transAxes,
                    ha='center', va='center')
        ax4.set_aspect('equal')
        if has_les:
            plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    plt.savefig('advanced_mf_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Diagnostic plots saved to 'advanced_mf_diagnostics.png'")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Advanced Multi-Fidelity Gaussian Process Regression")
    print("=" * 80)

    # Initialize and load data
    model = AdvancedMultiFidelityGPR(
        spatial_length_scale=50.0,
        angular_length_scale=10.0,
        use_matern=True
    )

    print("\nLoading data...")
    data = model.load_data()

    # Prepare training data with spatial subsampling
    print("\nPreparing training data...")
    X_hf, y_hf, y_lf = model.prepare_training_data(subsample_spatial=2)

    # Fit model
    print("\nFitting model...")
    model.fit(X_hf, y_hf, y_lf, method='autoregressive')

    # Generate predictions
    print("\nGenerating predictions...")
    test_angles = [15.0, 90.0, 180.0]
    create_diagnostic_plots(model, test_angles)

    # Save model
    print("\nSaving model...")
    model.save_model('mf_gpr_model.pkl')

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return model


if __name__ == "__main__":
    model = main()

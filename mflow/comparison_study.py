import numpy as np
import matplotlib.pyplot as plt
from multifidelity_gpr import MultiFidelityGPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')
#
# Comparison Study: Single-Fidelity vs Multi-Fidelity Approaches
#
def compare_approaches():
    """
    Compare RANS-only, LES-only, and multi-fidelity approaches.
    """
    print("\n" + "="*80)
    print("COMPARISON STUDY: Single-Fidelity vs Multi-Fidelity Approaches")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    mf_model = MultiFidelityGPR()
    x_grid, y_grid, data = mf_model.load_data('truncated_data')

    les_angles = data['les']['angles']
    les_data = data['les']['data']
    rans_data = data['rans']['data']

    # Select test angles (not in LES training set)
    test_angles = [10.0, 45.0, 100.0, 175.0, 200.0, 270.0]
    print(f"Test angles: {test_angles}")

    # Prepare data for multi-fidelity model
    print("\n1. Training Multi-Fidelity GPR...")
    X_hf, y_hf, y_lf = mf_model.prepare_training_data(data)

    # Subsample for demonstration
    n_sample = int(len(y_hf) * 0.05)
    indices = np.random.choice(len(y_hf), n_sample, replace=False)
    mf_model.fit(X_hf[indices], y_hf[indices], y_lf[indices])

    # Prepare LES-only GP (for comparison)
    print("\n2. Training LES-only GP...")
    kernel_les = C(1.0) * RBF([10.0, 50.0, 50.0]) + WhiteKernel(0.01)
    gp_les = GaussianProcessRegressor(kernel=kernel_les, n_restarts_optimizer=2,
                                      normalize_y=True, alpha=1e-6)
    gp_les.fit(X_hf[indices], y_hf[indices])

    # Comparison metrics
    results = {
        'rans': {'rmse': [], 'mae': [], 'cost': 'Low'},
        'les_only': {'rmse': [], 'mae': [], 'cost': 'High'},
        'multifidelity': {'rmse': [], 'mae': [], 'cost': 'Medium'}
    }

    print("\n3. Evaluating on test angles...")
    print("-" * 80)

    for test_angle in test_angles:
        # Find closest actual LES data for ground truth
        angle_diff = np.abs(les_angles - test_angle)
        closest_idx = angle_diff.argmin()
        closest_angle = les_angles[closest_idx]

        if angle_diff[closest_idx] > 5.0:
            print(f"Angle {test_angle}°: No nearby LES data (skipping)")
            continue

        # Ground truth
        les_field = les_data[closest_idx]
        mask = les_field > 1e-6

        if mask.sum() == 0:
            continue

        # Method 1: RANS only (simple interpolation)
        angle_idx = int(test_angle) % 360
        rans_field = rans_data[angle_idx]
        rans_rmse = np.sqrt(np.mean((rans_field[mask] - les_field[mask])**2))
        rans_mae = np.mean(np.abs(rans_field[mask] - les_field[mask]))

        # Method 2: LES-only GP
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
        X_test = np.column_stack([
            np.full(len(x_grid) * len(y_grid), closest_angle),
            X_mesh.ravel(),
            Y_mesh.ravel()
        ])
        les_pred = gp_les.predict(X_test).reshape(len(x_grid), len(y_grid))
        les_rmse = np.sqrt(np.mean((les_pred[mask] - les_field[mask])**2))
        les_mae = np.mean(np.abs(les_pred[mask] - les_field[mask]))

        # Method 3: Multi-fidelity GPR
        mf_pred, _ = mf_model.predict_field(closest_angle, data)
        mf_rmse = np.sqrt(np.mean((mf_pred[mask] - les_field[mask])**2))
        mf_mae = np.mean(np.abs(mf_pred[mask] - les_field[mask]))

        # Store results
        results['rans']['rmse'].append(rans_rmse)
        results['rans']['mae'].append(rans_mae)
        results['les_only']['rmse'].append(les_rmse)
        results['les_only']['mae'].append(les_mae)
        results['multifidelity']['rmse'].append(mf_rmse)
        results['multifidelity']['mae'].append(mf_mae)

        # Print comparison
        print(f"Angle {closest_angle:6.2f}°:")
        print(f"  RANS only:        RMSE={rans_rmse:.5f}  MAE={rans_mae:.5f}")
        print(f"  LES-only GP:      RMSE={les_rmse:.5f}  MAE={les_mae:.5f}")
        print(f"  Multi-fidelity:   RMSE={mf_rmse:.5f}  MAE={mf_mae:.5f}")
        print(f"  Improvement:      {(1 - mf_rmse/rans_rmse)*100:+.1f}% vs RANS, "
              f"{(1 - mf_rmse/les_rmse)*100:+.1f}% vs LES-only")
        print()
    #
    # Summary statistics
    #
    print("-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)

    for method in ['rans', 'les_only', 'multifidelity']:
        if results[method]['rmse']:
            avg_rmse = np.mean(results[method]['rmse'])
            avg_mae = np.mean(results[method]['mae'])
            std_rmse = np.std(results[method]['rmse'])

            method_name = method.replace('_', ' ').title()
            print(f"{method_name:20s}: RMSE = {avg_rmse:.5f} ± {std_rmse:.5f}, "
                  f"MAE = {avg_mae:.5f}, Cost = {results[method]['cost']}")

    # Calculate improvements
    if results['rans']['rmse'] and results['multifidelity']['rmse']:
        improvement_rans = (1 - np.mean(results['multifidelity']['rmse']) /
                           np.mean(results['rans']['rmse'])) * 100
        improvement_les = (1 - np.mean(results['multifidelity']['rmse']) /
                          np.mean(results['les_only']['rmse'])) * 100

        print(f"\nMulti-fidelity improvement:")
        print(f"  {improvement_rans:+.1f}% better than RANS-only")
        print(f"  {improvement_les:+.1f}% better than LES-only GP")

    # Visualization
    print("\n4. Creating comparison visualization...")
    create_comparison_plot(mf_model, gp_les, data, test_angles[0])    


def create_comparison_plot(mf_model, gp_les, data, test_angle):
    """
    Create side-by-side comparison visualization.
    """
    x_grid = data['grid']['x']
    y_grid = data['grid']['y']

    # Find closest LES angle
    les_angles = data['les']['angles']
    closest_idx = np.argmin(np.abs(les_angles - test_angle))
    actual_angle = les_angles[closest_idx]

    # Get predictions
    rans_field = data['rans']['data'][int(actual_angle) % 360]

    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_test = np.column_stack([
        np.full(len(x_grid) * len(y_grid), actual_angle),
        X_mesh.ravel(),
        Y_mesh.ravel()
    ])

    les_pred = gp_les.predict(X_test).reshape(len(x_grid), len(y_grid))
    mf_pred, mf_std = mf_model.predict_field(actual_angle, data)

    # Ground truth
    les_truth = data['les']['data'][closest_idx]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmin, vmax = 0, 0.6

    # Row 1: Predictions
    im1 = axes[0, 0].contourf(x_grid, y_grid, rans_field.T, levels=20,
                              vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0, 0].set_title(f'RANS Only\n(Low-Fidelity)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].contourf(x_grid, y_grid, les_pred.T, levels=20,
                              vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0, 1].set_title(f'LES-Only GP\n(High-Fidelity Only)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].contourf(x_grid, y_grid, mf_pred.T, levels=20,
                              vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0, 2].set_title(f'Multi-Fidelity GPR\n(Combined)')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2])

    # Row 2: Errors vs ground truth
    mask = les_truth > 1e-6
    error_rans = np.abs(rans_field - les_truth)
    error_rans[~mask] = 0

    error_les = np.abs(les_pred - les_truth)
    error_les[~mask] = 0

    error_mf = np.abs(mf_pred - les_truth)
    error_mf[~mask] = 0

    vmax_err = max(error_rans.max(), error_les.max(), error_mf.max())

    im4 = axes[1, 0].contourf(x_grid, y_grid, error_rans.T, levels=20,
                              vmin=0, vmax=vmax_err, cmap='Reds')
    axes[1, 0].set_title(f'Error vs LES Truth')
    axes[1, 0].set_aspect('equal')
    rmse_rans = np.sqrt(np.mean(error_rans[mask]**2))
    axes[1, 0].text(0.5, 0.02, f'RMSE={rmse_rans:.4f}',
                    transform=axes[1, 0].transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].contourf(x_grid, y_grid, error_les.T, levels=20,
                              vmin=0, vmax=vmax_err, cmap='Reds')
    axes[1, 1].set_title(f'Error vs LES Truth')
    axes[1, 1].set_aspect('equal')
    rmse_les = np.sqrt(np.mean(error_les[mask]**2))
    axes[1, 1].text(0.5, 0.02, f'RMSE={rmse_les:.4f}',
                    transform=axes[1, 1].transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].contourf(x_grid, y_grid, error_mf.T, levels=20,
                              vmin=0, vmax=vmax_err, cmap='Reds')
    axes[1, 2].set_title(f'Error vs LES Truth')
    axes[1, 2].set_aspect('equal')
    rmse_mf = np.sqrt(np.mean(error_mf[mask]**2))
    axes[1, 2].text(0.5, 0.02, f'RMSE={rmse_mf:.4f}',
                    transform=axes[1, 2].transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im6, ax=axes[1, 2])

    plt.suptitle(f'Method Comparison at {actual_angle:.1f}°', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_study.png', dpi=150, bbox_inches='tight')
    print("   Comparison plot saved to 'comparison_study.png'")


if __name__ == "__main__":
    compare_approaches()

import numpy as np
import matplotlib.pyplot as plt
from multifidelity_gpr import MultiFidelityGPR
#
# QUICK DEMO CPU (Not recommended!)
#
def quick_demo():
    """
    Quick demonstration of the multi-fidelity GPR model.
    """
    print("\n" + "="*70)
    print("QUICK START: Multi-Fidelity Gaussian Process Regression")
    print("="*70 + "\n")

    # Initialize model
    print("Step 1: Initializing model...")
    model = MultiFidelityGPR(kernel_length_scale=15.0, noise_level=0.01)

    # Load data
    print("\nStep 2: Loading data...")
    x_grid, y_grid, data = model.load_data('truncated_data')

    # Prepare training data (with subsampling for speed)
    print("\nStep 3: Preparing training data...")
    X_hf, y_hf, y_lf_at_hf = model.prepare_training_data(data)

    # Subsample to 5% for quick demo
    print("   Subsampling to 5% for quick demonstration...")
    n_total = len(y_hf)
    n_sample = int(n_total * 0.05)
    indices = np.random.choice(n_total, n_sample, replace=False)
    X_hf_sub = X_hf[indices]
    y_hf_sub = y_hf[indices]
    y_lf_sub = y_lf_at_hf[indices]
    print(f"   Using {n_sample} of {n_total} training points")

    # Train model
    print("\nStep 4: Training model...")
    model.fit(X_hf_sub, y_hf_sub, y_lf_sub)

    # Make predictions at a few test angles
    print("\nStep 5: Making predictions...")
    test_angles = [30.0, 90.0, 180.0, 270.0]

    # Validation: compare with actual LES data
    print("\nStep 6: Validating against LES data...")
    les_angles = data['les']['angles']
    les_data = data['les']['data']

    # Find LES angles close to test angles
    validation_results = []
    for test_angle in test_angles:
        # Find closest LES angle
        angle_diff = np.abs(les_angles - test_angle)
        if angle_diff.min() < 1.0:  # Within 1 degrees
            les_idx = angle_diff.argmin()
            actual_angle = les_angles[les_idx]

            # Get prediction
            pred, std = model.predict_field(actual_angle, data)
            les_field = les_data[les_idx]

            # Compute error on non-zero regions
            mask = les_field > 1e-6
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((pred[mask] - les_field[mask])**2))
                mae = np.mean(np.abs(pred[mask] - les_field[mask]))
                rel_error = rmse / np.mean(les_field[mask])

                validation_results.append({
                    'angle': actual_angle,
                    'rmse': rmse,
                    'mae': mae,
                    'rel_error': rel_error
                })

                print(f"   Angle {actual_angle:6.2f}°: RMSE={rmse:.5f}, "
                      f"MAE={mae:.5f}, Relative Error={rel_error*100:.2f}%")

    # Create visualization
    print("\nStep 7: Creating visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i, angle in enumerate(test_angles):
        # Get predictions
        pred, std = model.predict_field(angle, data)

        # Get RANS for comparison
        angle_idx = int(angle) % 360
        rans_field = data['rans']['data'][angle_idx]

        # Plot RANS
        ax = axes[0, i]
        im = ax.contourf(x_grid, y_grid, rans_field.T, levels=20,
                        vmin=0, vmax=0.6, cmap='viridis')
        ax.set_title(f'RANS - {angle}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot MF-GPR prediction
        ax = axes[1, i]
        im = ax.contourf(x_grid, y_grid, pred.T, levels=20,
                        vmin=0, vmax=0.6, cmap='viridis')
        ax.set_title(f'MF-GPR - {angle}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot uncertainty
        ax = axes[2, i]
        im = ax.contourf(x_grid, y_grid, std.T, levels=20, cmap='Reds')
        ax.set_title(f'Uncertainty - {angle}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Multi-Fidelity GPR: Wind Field Predictions', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('quick_start_results.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved to 'quick_start_results.png'")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model trained on {n_sample} points (5% subsample)")
    print(f"Scaling factor ρ = {model.rho:.4f}")
    print(f"GP kernel: {model.gp_delta.kernel_}")

    if validation_results:
        avg_rmse = np.mean([r['rmse'] for r in validation_results])
        avg_rel_error = np.mean([r['rel_error'] for r in validation_results])
        print(f"\nValidation metrics:")
        print(f"  Average RMSE: {avg_rmse:.5f}")
        print(f"  Average relative error: {avg_rel_error*100:.2f}%")

    print("\nNote: For better accuracy, train on more data points by:")
    print("  1. Reducing subsample_ratio in multifidelity_gpr.py")
    print("  2. Or using prepare_training_data without subsampling")

    # Save the trained model
    print("\nStep 8: Saving model...")
    model.save_model('mf_gpr_model_cpu.pkl')
    print("   Model saved to 'mf_gpr_model_cpu.pkl'")

    print("\n" + "="*70)
    print("Quick start complete! Check 'quick_start_results.png' for results.")
    print("="*70 + "\n")

    return model, data


if __name__ == "__main__":
    model, data = quick_demo()

    print("\nYou can now use the trained model:")
    print("  >>> prediction, uncertainty = model.predict_field(45.0, data)")
    print("  >>> # Predict at any angle between 0-360 degrees")

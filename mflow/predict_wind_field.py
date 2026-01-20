import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os


def load_model_and_data(model_path: str, use_gpu: bool = False):
    """
        Load the trained model and data.

        Parameters
        ----------
        model_path : str
            Path to the saved model file
        use_gpu : bool
            Whether to use GPU acceleration

        Returns
        -------
        model : MultiFidelityGPR or MultiFidelityGPR_GPU
            Loaded model
        data : dict
            Data dictionary with RANS and grid information
    """
    print(f"Loading model from {model_path}...")

    # Determine model type from file extension
    is_gpu_model = model_path.endswith('.pth')

    if is_gpu_model or use_gpu:
        try:
            from multifidelity_gpr_gpu import MultiFidelityGPR_GPU
            device = 'cuda' if use_gpu else 'cpu'
            model = MultiFidelityGPR_GPU(device=device)
            model.load_model(model_path)
            print(f"GPU model loaded successfully (device: {device})")
        except ImportError:
            print("Error: PyTorch/GPyTorch not installed. Install with:")
            print("  pip install torch gpytorch")
            sys.exit(1)
    else:
        from multifidelity_gpr import MultiFidelityGPR
        model = MultiFidelityGPR()
        model.load_model(model_path)
        print("CPU model loaded successfully")

    # Load data
    print("Loading wind field data...")
    x_grid, y_grid, data = model.load_data('truncated_data')
    print(f"Data loaded: {len(x_grid)} x {len(y_grid)} grid points")

    return model, data, x_grid, y_grid


def predict_at_angles(model, data, x_grid, y_grid, angles: List[float],
                      save_prefix: str = 'prediction'):
    """
        Make predictions at specified angles.

        Parameters
        ----------
        model : MultiFidelityGPR or MultiFidelityGPR_GPU
            Trained model
        data : dict
            Data dictionary
        x_grid : np.ndarray
            X coordinates
        y_grid : np.ndarray
            Y coordinates
        angles : list of float
            Angles to predict at (in degrees)
        save_prefix : str
            Prefix for saved files

        Returns
        -------
        predictions : dict
            Dictionary mapping angles to (prediction, uncertainty) tuples
    """
    predictions = {}

    print(f"\nMaking predictions at {len(angles)} angle(s)...")

    for angle in angles:
        print(f"  Predicting at angle {angle:.2f}°...", end=' ')

        try:
            pred, std = model.predict_field(angle, data)
            predictions[angle] = (pred, std)

            # Basic statistics
            valid_mask = np.isfinite(pred) & (pred > 1e-6)
            if valid_mask.sum() > 0:
                mean_pred = np.mean(pred[valid_mask])
                max_pred = np.max(pred[valid_mask])
                mean_uncertainty = np.mean(std[valid_mask])
                print(f"✓ (mean={mean_pred:.4f}, max={max_pred:.4f}, unc={mean_uncertainty:.4f})")
            else:
                print("✓ (no valid points)")

        except Exception as e:
            print(f"✗ Error: {e}")
            predictions[angle] = (None, None)

    return predictions


def visualize_predictions(predictions: dict, x_grid, y_grid, data,
                         save_filename: str = 'predictions.png'):
    """
        Create visualization of predictions.

        Parameters
        ----------
        predictions : dict
            Dictionary mapping angles to (prediction, uncertainty) tuples
        x_grid : np.ndarray
            X coordinates
        y_grid : np.ndarray
            Y coordinates
        data : dict
            Data dictionary (for RANS comparison)
        save_filename : str
            Filename to save the plot
    """
    n_angles = len(predictions)
    if n_angles == 0:
        print("No predictions to visualize")
        return

    # Get LES angles for comparison
    les_angles = data['les']['angles']

    # Determine which predictions have LES data available
    has_les = {}
    les_indices = {}
    for angle in predictions.keys():
        angle_diff = np.abs(les_angles - angle)
        if angle_diff.min() < 0.5:  # Within 0.5 degrees
            les_idx = angle_diff.argmin()
            has_les[angle] = True
            les_indices[angle] = les_idx
        else:
            has_les[angle] = False

    # Determine number of rows (add LES row if any angle has LES data)
    any_has_les = any(has_les.values())
    n_rows = 4 if any_has_les else 3

    # Create figure
    fig, axes = plt.subplots(n_rows, min(n_angles, 4), figsize=(4*min(n_angles, 4), 4*n_rows))

    if n_angles == 1:
        axes = axes.reshape(-1, 1)

    angles_list = sorted(predictions.keys())[:4]  # Limit to 4 for visualization

    for i, angle in enumerate(angles_list):
        pred, std = predictions[angle]

        if pred is None:
            continue

        # Get RANS for comparison
        angle_idx = int(angle) % 360
        rans_field = data['rans']['data'][angle_idx]

        # Row index counter
        row = 0

        # Plot RANS
        ax = axes[row, i]
        im = ax.contourf(x_grid, y_grid, rans_field.T, levels=20,
                        vmin=0, vmax=0.6, cmap='viridis')
        ax.set_title(f'RANS - {angle:.1f}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        row += 1

        # Plot LES if available
        if any_has_les:
            ax = axes[row, i]
            if has_les[angle]:
                les_idx = les_indices[angle]
                les_field = data['les']['data'][les_idx]
                actual_les_angle = les_angles[les_idx]
                im = ax.contourf(x_grid, y_grid, les_field.T, levels=20,
                                vmin=0, vmax=0.6, cmap='viridis')
                ax.set_title(f'LES - {actual_les_angle:.2f}°', fontsize=10)
                ax.set_aspect('equal')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # No LES data available for this angle
                ax.text(0.5, 0.5, 'No LES data\navailable',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.set_aspect('equal')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_xticks([])
                ax.set_yticks([])
            row += 1

        # Plot MF-GPR prediction
        ax = axes[row, i]
        im = ax.contourf(x_grid, y_grid, pred.T, levels=20,
                        vmin=0, vmax=0.6, cmap='viridis')
        ax.set_title(f'MF-GPR Prediction - {angle:.1f}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        row += 1

        # Plot uncertainty
        ax = axes[row, i]
        im = ax.contourf(x_grid, y_grid, std.T, levels=20, cmap='Reds')
        ax.set_title(f'Uncertainty - {angle:.1f}°', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Multi-Fidelity GPR Wind Field Predictions', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to '{save_filename}'")


def save_predictions_to_file(predictions: dict, x_grid, y_grid,
                             save_filename: str = 'predictions.npz'):
    """
        Save predictions to a NumPy compressed file.

        Parameters
        ----------
        predictions : dict
            Dictionary mapping angles to (prediction, uncertainty) tuples
        x_grid : np.ndarray
            X coordinates
        y_grid : np.ndarray
            Y coordinates
        save_filename : str
            Filename to save predictions
    """
    # Prepare data for saving
    save_dict = {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'angles': np.array(list(predictions.keys()))
    }

    for i, (angle, (pred, std)) in enumerate(predictions.items()):
        if pred is not None:
            save_dict[f'prediction_{i}'] = pred
            save_dict[f'uncertainty_{i}'] = std

    np.savez_compressed(save_filename, **save_dict)
    print(f"Predictions saved to '{save_filename}'")
    print(f"Load with: data = np.load('{save_filename}')")

def usage_help():
    print("""
            Wind Field Prediction Script

            This script loads a pre-trained multi-fidelity GPR model and makes predictions
            at specified wind angles. It can work with both CPU and GPU models.

            Usage:
                python predict_wind_field.py --model mf_gpr_model_cpu.pkl --angle 45.5
                python predict_wind_field.py --model mf_gpr_model_gpu.pth --angles 30 60 90 --use-gpu
                python predict_wind_field.py --model mf_gpr_model_cpu.pkl --angle-range 0 360 10
            """)

def main():
    parser = argparse.ArgumentParser(
        description='Predict wind fields using trained multi-fidelity GPR model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Run 'python predict_wind_field.py --usage_help' for usage instructions."""
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model file (.pkl for CPU, .pth for GPU)')
    parser.add_argument('--angle', type=float,
                       help='Single angle to predict (degrees)')
    parser.add_argument('--angles', type=float, nargs='+',
                       help='Multiple angles to predict (degrees)')
    parser.add_argument('--angle-range', type=float, nargs=3,
                       metavar=('START', 'STOP', 'STEP'),
                       help='Angle range: start stop step (degrees)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--output', type=str, default='predictions.png',
                       help='Output filename for visualization (default: predictions.png)')
    parser.add_argument('--save-data', type=str,
                       help='Save predictions to .npz file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--usage_help', action='store_true',
                       help='Show detailed usage information and exit')

    args = parser.parse_args()

    # Determine angles to predict
    angles = []
    if args.angle is not None:
        angles = [args.angle]
    elif args.angles is not None:
        angles = args.angles
    elif args.angle_range is not None:
        start, stop, step = args.angle_range
        angles = np.arange(start, stop, step).tolist()
    else:
        print("Error: Must specify --angle, --angles, or --angle-range")
        parser.print_help()
        sys.exit(1)

    if args.usage_help:
        usage_help()
        sys.exit(0)

    print("="*70)
    print("Multi-Fidelity GPR Wind Field Prediction")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Angles: {len(angles)} angles from {min(angles):.1f}° to {max(angles):.1f}°")
    print("="*70)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)

    # Load model and data
    model, data, x_grid, y_grid = load_model_and_data(args.model, args.use_gpu)

    # Make predictions
    predictions = predict_at_angles(model, data, x_grid, y_grid, angles)

    # Visualize
    if not args.no_plot:
        visualize_predictions(predictions, x_grid, y_grid, data, args.output)

    # Save data if requested
    if args.save_data:
        save_predictions_to_file(predictions, x_grid, y_grid, args.save_data)

    print("\n" + "="*70)
    print("Prediction complete!")
    print("="*70)


if __name__ == "__main__":
    main()

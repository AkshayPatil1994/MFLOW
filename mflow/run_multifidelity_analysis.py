import sys
import argparse
import time
#
# BANNER
#
def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print(" "*20 + "MULTI-FIDELITY GAUSSIAN PROCESS REGRESSION")
    print(" "*25 + "Wind Field Prediction System")
    print("="*80)
    print("\nCombining RANS (low-fidelity) and LES (high-fidelity) data")
    print("for accurate wind prediction at arbitrary angles\n")
#
# HOW TO USE THE MODEL 
#
def how_to_use_model():
    """
        Print the full set of instructions on how to use the model for training on a dataset
    """

    print("\n" + "="*80)
    print("HOW TO USE THE MULTI-FIDELITY GPR MODEL")
    print("="*80)
    print("""Primary Script: Multi-Fidelity GPR Analysis Pipeline

    This script provides a complete workflow for multi-fidelity wind prediction:
    1. Data exploration and visualization
    2. Model training and validation
    3. Comparison with baseline methods
    4. Results export and diagnostics

    Usage:
        python run_multifidelity_analysis.py [options]

    Main Options:
        --explore-only             : Only run data exploration
        --quick                    : Quick demo with subsampling
        --full                     : Full analysis with all data
        --compare                  : Run comparison study
        --advanced                 : Use advanced MF-GPR model

    GPU Options:
        --use-gpu                  : Use GPU acceleration (requires GPU and PyTorch/GPyTorch)
        --device {auto,cuda,cpu}   : Device to use (default: 'auto')

    Batch Training Options (GPU only):
        --use-batch-training       : Enable mini-batch training (reduces memory usage)
        --batch-size N             : Batch size for training (default: 10000)
        --training-iter N          : Number of training iterations (default: 50)

    Model Hyperparameters:
        --kernel-length-scale F    : Kernel length scale (default: 15.0)
        --noise-level F            : Noise level (default: 0.01)
        --max-training-samples N   : Max training samples (default: auto)
        --spatial-length-scale F   : Spatial length scale for advanced model (default: 50.0)
        --angular-length-scale F   : Angular length scale for advanced model (default: 10.0)
        --use-matern               : Use Matern kernel for advanced model (default: True)

    Data Options:
        --subsample-ratio F        : Ratio of data to use (default: 0.2 = 20%)
        --subsample-spatial N      : Spatial subsampling factor for advanced model

    Model Saving Options:
        --save-model [PATH]        : Save trained model (default: surrogate_model/mflow.pkl)
        --no-save-model            : Disable automatic model saving
    """)
#
# DATA EXPLORATION 
#
def run_data_exploration():
    """Run data exploration and visualization."""
    print("\n" + "─"*80)
    print("STEP 1: DATA EXPLORATION")
    print("─"*80)

    try:
        from visualize_data import explore_data
        explore_data()
        print("Data exploration complete...")
        return True
    except Exception as e:
        print(f"Error in data exploration: {e}...")
        return False
#
# QUICK DEMONSTRATION
#
def run_quick_demo(use_gpu=False, device='auto', args=None):
    """
        Run quick demonstration with subsampled data. Defaults to CPU if GPU not specified.
    """
    print("\n" + "─"*80)
    print("QUICK DEMONSTRATION")
    if use_gpu:
        print("(GPU-Accelerated)")
    if args and args.use_batch_training:
        print(f"(Batch Mode: batch_size={args.batch_size}, iter={args.training_iter})")
    print("─"*80)

    try:
        if use_gpu:
            from quick_start_gpu import quick_demo_gpu
            model, data = quick_demo_gpu(device=device)
        else:
            from quick_start import quick_demo
            model, data = quick_demo()
        print("Quick demo complete...")
        return True, model, data
    except Exception as e:
        print(f"Error in quick demo: {e}...")
        import traceback
        traceback.print_exc()
        return False, None, None
#
# RUN FULL TRAINING
#
def run_full_training(use_gpu=False, device='auto', args=None):
    """
        Run full model training with all data.
    """

    print("\n" + "─"*80)
    print("FULL MODEL TRAINING")
    if use_gpu:
        print("(GPU-Accelerated)")
    if args and args.use_batch_training:
        print(f"(Batch Mode: batch_size={args.batch_size}, iter={args.training_iter})")
    print("─"*80)

    try:
        import numpy as np

        # Get parameters from args or use defaults
        kernel_length_scale = args.kernel_length_scale if args else 15.0
        noise_level = args.noise_level if args else 0.01
        max_training_samples = args.max_training_samples if args else 30000
        training_iter = args.training_iter if args else 50
        use_batch_training = args.use_batch_training if args else False
        batch_size = args.batch_size if args else 10000
        subsample_ratio = args.subsample_ratio if args else 0.2

        if use_gpu:
            from multifidelity_gpr_gpu import MultiFidelityGPR_GPU
            print("Initializing GPU-accelerated model...")
            print(f"  Kernel length scale: {kernel_length_scale}")
            print(f"  Noise level: {noise_level}")
            if not use_batch_training and max_training_samples:
                print(f"  Max training samples: {max_training_samples:,}")

            model = MultiFidelityGPR_GPU(
                kernel_length_scale=kernel_length_scale,
                noise_level=noise_level,
                device=device,
                max_training_samples=max_training_samples if not use_batch_training else None
            )
        else:
            from multifidelity_gpr import MultiFidelityGPR
            print("Initializing model...")
            model = MultiFidelityGPR(kernel_length_scale=kernel_length_scale,
                                    noise_level=noise_level)

        print("Loading data...")
        x_grid, y_grid, data = model.load_data('truncated_data')

        print("Preparing training data...")
        X_hf, y_hf, y_lf = model.prepare_training_data(data)

        # Handle subsampling if not using batch training
        if use_batch_training:
            print(f"Using ALL {len(y_hf):,} points with batch training")
            X_train = X_hf
            y_train = y_hf
            y_lf_train = y_lf
        else:
            n_sample = int(len(y_hf) * subsample_ratio)
            print(f"Using {n_sample:,} of {len(y_hf):,} points ({subsample_ratio*100:.0f}%)")
            indices = np.random.choice(len(y_hf), n_sample, replace=False)
            X_train = X_hf[indices]
            y_train = y_hf[indices]
            y_lf_train = y_lf[indices]

        print("Training model (this may take a few minutes)...")
        print(f"  Training iterations: {training_iter}")
        if use_batch_training:
            print(f"  Batch size: {batch_size:,}")

        start_time = time.time()

        if use_gpu:
            model.fit(X_train, y_train, y_lf_train,
                     training_iter=training_iter,
                     use_batch_training=use_batch_training,
                     batch_size=batch_size)
        else:
            model.fit(X_train, y_train, y_lf_train)

        training_time = time.time() - start_time

        print(f"Training complete in {training_time:.1f} seconds...")

        # Quick validation
        print("\nValidating predictions...")
        les_angles = data['les']['angles'][:3]  
        les_data_arr = data['les']['data']

        rmse_scores = []
        for i, angle in enumerate(les_angles):
            pred, _ = model.predict_field(angle, data)
            les_field = les_data_arr[i]
            mask = les_field > 1e-6

            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((pred[mask] - les_field[mask])**2))
                rmse_scores.append(rmse)
                print(f"  Angle {angle:.2f}°: RMSE = {rmse:.5f}")

        if rmse_scores:
            print(f"\nMean RMSE: {np.mean(rmse_scores):.5f}")

        # Save model if requested
        if args and args.save_model and not args.no_save_model:
            print(f"\nSaving model to {args.save_model}...")
            import os
            save_dir = os.path.dirname(args.save_model)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"  Created directory: {save_dir}")

            # Determine extension based on GPU/CPU
            if use_gpu and not args.save_model.endswith('.pth'):
                save_path = args.save_model.replace('.pkl', '.pth')
                print(f"  Using .pth extension for GPU model: {save_path}")
            else:
                save_path = args.save_model

            model.save_model(save_path)
            print(f"Model saved successfully...")

        return True, model, data

    except Exception as e:
        print(f"Error in full training: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None
#
# COMPARISON STUDY
#
def run_comparison_study():
    """
        Run comparison study between methods.
    """
    print("\n" + "─"*80)
    print("COMPARISON STUDY")
    print("─"*80)

    try:
        from comparison_study import compare_approaches
        compare_approaches()
        print("Comparison study complete...")
        return True
    except Exception as e:
        print(f"Error in comparison study: {e}")
        import traceback
        traceback.print_exc()
        return False
#
# ADVANCED MODEL
#
def run_advanced_model(use_gpu=False, device='auto', args=None):
    """Run advanced multi-fidelity model."""
    print("\n" + "─"*80)
    print("ADVANCED MODEL")
    if use_gpu:
        print("(GPU-Accelerated)")
    if args and args.use_batch_training:
        print(f"(Batch Mode: batch_size={args.batch_size}, iter={args.training_iter})")
    print("─"*80)

    try:
        # Get parameters from args or use defaults
        spatial_length_scale = args.spatial_length_scale if args else 50.0
        angular_length_scale = args.angular_length_scale if args else 10.0
        use_matern = args.use_matern if args else True
        subsample_spatial = args.subsample_spatial if args else None

        if use_gpu:
            from advanced_multifidelity_gpr_gpu import AdvancedMultiFidelityGPR_GPU
            print("Initializing advanced GPU-accelerated model...")
            print(f"  Spatial length scale: {spatial_length_scale}")
            print(f"  Angular length scale: {angular_length_scale}")
            print(f"  Kernel: {'Matern' if use_matern else 'RBF'}")
            if subsample_spatial:
                print(f"  Spatial subsampling: {subsample_spatial}")

            model = AdvancedMultiFidelityGPR_GPU(
                spatial_length_scale=spatial_length_scale,
                angular_length_scale=angular_length_scale,
                use_matern=use_matern,
                device=device
            )
            # Load and prepare data
            data = model.load_data('truncated_data')
            X_hf, y_hf, y_lf = model.prepare_training_data(
                subsample_spatial=subsample_spatial if subsample_spatial else 2
            )

            # Fit model
            training_iter = args.training_iter if args else 50
            model.fit(X_hf, y_hf, y_lf, method='autoregressive',
                     training_iter=training_iter)
        else:
            from advanced_multifidelity_gpr import main as advanced_main
            model = advanced_main()

        # Save model if requested
        if args and args.save_model and not args.no_save_model:
            print(f"\nSaving advanced model to {args.save_model}...")
            import os
            save_dir = os.path.dirname(args.save_model)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"  Created directory: {save_dir}")

            # Use _advanced suffix and appropriate extension
            base_path = args.save_model.replace('.pkl', '').replace('.pth', '')
            if use_gpu:
                save_path = f"{base_path}_advanced.pth"
            else:
                save_path = f"{base_path}_advanced.pkl"

            print(f"  Saving to: {save_path}")
            model.save_model(save_path)
            print(f"Advanced model saved successfully...")

        print("Advanced model complete...")
        return True, model
    except Exception as e:
        print(f"Error in advanced model: {e}")
        import traceback
        traceback.print_exc()
        return False, None
#
# PRINT SUMMARY 
#
def print_summary(results):
    """Print summary of completed steps."""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    steps = [
        ("Data Exploration", results.get('explore', False)),
        ("Quick Demo", results.get('quick', False)),
        ("Full Training", results.get('full', False)),
        ("Comparison Study", results.get('compare', False)),
        ("Advanced Model", results.get('advanced', False)),
    ]

    for step_name, completed in steps:
        status = "✓" if completed else "─"
        print(f"{status} {step_name}")

    print("\nGenerated files:")
    files = [
        ("data_exploration.png", "Data visualization and statistics"),
        ("quick_start_results.png", "Quick demo predictions"),
        ("multifidelity_predictions.png", "Full model predictions"),
        ("comparison_study.png", "Method comparison"),
        ("advanced_mf_diagnostics.png", "Advanced model diagnostics"),
        ("mf_gpr_model.pkl", "Saved model (for reuse)"),
    ]

    import os
    for filename, description in files:
        if os.path.exists(filename):
            print(f"{filename:<35s} - {description}")

    print("="*80 + "\n")
#
# DEFINE MAIN
#
def main():
    """
        Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Fidelity GPR Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multifidelity_analysis.py --explore-only
  python run_multifidelity_analysis.py --quick
  python run_multifidelity_analysis.py --full --compare
  python run_multifidelity_analysis.py --advanced
        """
    )

    parser.add_argument('--help', action='store_true',
                       help='Prints full instructions on how to use the model')
    parser.add_argument('--explore-only', action='store_true',
                       help='Only run data exploration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo with subsampling')
    parser.add_argument('--full', action='store_true',
                       help='Full analysis with more data')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison study')
    parser.add_argument('--advanced', action='store_true',
                       help='Use advanced MF-GPR model')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration (requires PyTorch and GPyTorch)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for computation (default: auto)')

    # Batch training parameters
    parser.add_argument('--use-batch-training', action='store_true',
                       help='Use mini-batch training for large datasets (reduces memory usage)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size for batch training (default: 10000)')
    parser.add_argument('--training-iter', type=int, default=50,
                       help='Number of training iterations (default: 50)')

    # Model hyperparameters
    parser.add_argument('--kernel-length-scale', type=float, default=15.0,
                       help='Kernel length scale for basic model (default: 15.0)')
    parser.add_argument('--noise-level', type=float, default=0.01,
                       help='Noise level for model (default: 0.01)')
    parser.add_argument('--max-training-samples', type=int, default=None,
                       help='Maximum training samples (default: auto based on GPU memory)')

    # Advanced model parameters
    parser.add_argument('--spatial-length-scale', type=float, default=50.0,
                       help='Spatial length scale for advanced model (default: 50.0)')
    parser.add_argument('--angular-length-scale', type=float, default=10.0,
                       help='Angular length scale for advanced model (default: 10.0)')
    parser.add_argument('--use-matern', action='store_true', default=True,
                       help='Use Matern kernel for advanced model (default: True)')

    # Data parameters
    parser.add_argument('--subsample-ratio', type=float, default=0.2,
                       help='Ratio of data to use for training (default: 0.2 = 20%%)')
    parser.add_argument('--subsample-spatial', type=int, default=None,
                       help='Spatial subsampling factor for advanced model (default: None)')

    # Model saving parameters
    parser.add_argument('--save-model', type=str, nargs='?', const='surrogate_model/mflow.pkl',
                       help='Save trained model to file (default: surrogate_model/mflow.pkl if flag provided)')
    parser.add_argument('--no-save-model', action='store_true',
                       help='Disable automatic model saving')

    args = parser.parse_args()

    # If no arguments, run quick demo
    if not any(vars(args).values()):
        args.quick = True

    print_banner()

    if args.help:
        how_to_use_model()
        sys.exit(0)

    # Check GPU availability if requested
    if args.use_gpu:
        try:
            import torch
            if args.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = args.device

            if device == 'cuda' and not torch.cuda.is_available():
                print("WARNING: GPU requested but CUDA not available. Falling back to CPU.")
                device = 'cpu'
                args.use_gpu = False

            if args.use_gpu:
                print(f"\nGPU Acceleration Enabled: {device}")
                if device == 'cuda':
                    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
                print()
        except ImportError:
            print("WARNING: PyTorch not installed. GPU acceleration disabled.")
            print("Install with: pip install torch gpytorch\n")
            args.use_gpu = False
            device = 'cpu'
    else:
        device = 'cpu'

    results = {}
    start_time = time.time()

    # Step 1: Data exploration (always run unless full/advanced only)
    if not (args.full or args.advanced) or args.explore_only:
        results['explore'] = run_data_exploration()

    if args.explore_only:
        print_summary(results)
        return

    # Step 2: Quick demo
    if args.quick:
        success, model, data = run_quick_demo(use_gpu=args.use_gpu, device=device, args=args)
        results['quick'] = success

    # Step 3: Full training
    if args.full:
        success, model, data = run_full_training(use_gpu=args.use_gpu, device=device, args=args)
        results['full'] = success

    # Step 4: Comparison study
    if args.compare:
        results['compare'] = run_comparison_study()

    # Step 5: Advanced model
    if args.advanced:
        success, model = run_advanced_model(use_gpu=args.use_gpu, device=device, args=args)
        results['advanced'] = success

    # Print summary
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print_summary(results)
#
# RUN MAIN
#     
if __name__ == "__main__":
    main()

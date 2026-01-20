import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def compare_training(n_samples=5000):
    """Compare training time between CPU and GPU."""
    print("=" * 70)
    print(f"TRAINING COMPARISON (n={n_samples:,} samples)")
    print("=" * 70)

    # Load data once
    print("\nLoading data...")
    from multifidelity_gpr import MultiFidelityGPR
    model_temp = MultiFidelityGPR()
    x_grid, y_grid, data = model_temp.load_data('truncated_data')
    X_hf, y_hf, y_lf = model_temp.prepare_training_data(data)

    # Subsample
    n_total = len(y_hf)
    if n_samples > n_total:
        n_samples = n_total
        print(f"⚠ Requested {n_samples} samples but only {n_total} available")

    indices = np.random.choice(n_total, n_samples, replace=False)
    X_train = X_hf[indices]
    y_train = y_hf[indices]
    y_lf_train = y_lf[indices]

    print(f"Training dataset: {n_samples:,} points")

    # CPU Training
    print("\n1. CPU Training (scikit-learn)")
    print("   " + "-" * 40)
    try:
        from multifidelity_gpr import MultiFidelityGPR

        model_cpu = MultiFidelityGPR(kernel_length_scale=15.0, noise_level=0.01)
        model_cpu.load_data('truncated_data')

        start_time = time.time()
        model_cpu.fit(X_train, y_train, y_lf_train)
        cpu_time = time.time() - start_time

        print(f"   ✓ Training completed")
        print(f"   ✓ Time: {format_time(cpu_time)}")
        print(f"   ✓ Rho: {model_cpu.rho:.4f}")

        cpu_success = True

    except Exception as e:
        print(f"   ✗ CPU training failed: {e}")
        cpu_time = None
        cpu_success = False

    # GPU Training
    print("\n2. GPU Training (GPyTorch)")
    print("   " + "-" * 40)

    try:
        import torch
        if not torch.cuda.is_available():
            print("   ✗ CUDA not available - skipping GPU test")
            print("   ℹ Install with: pip install torch gpytorch")
            gpu_success = False
            gpu_time = None
        else:
            from multifidelity_gpr_gpu import MultiFidelityGPR_GPU

            model_gpu = MultiFidelityGPR_GPU(
                kernel_length_scale=15.0,
                noise_level=0.01,
                device='cuda'
            )
            model_gpu.load_data('truncated_data')

            start_time = time.time()
            model_gpu.fit(X_train, y_train, y_lf_train, training_iter=50)
            gpu_time = time.time() - start_time

            print(f"   ✓ Training completed")
            print(f"   ✓ Time: {format_time(gpu_time)}")
            print(f"   ✓ Rho: {model_gpu.rho:.4f}")
            print(f"   ✓ Device: {torch.cuda.get_device_name(0)}")

            gpu_success = True

    except ImportError:
        print("   ✗ GPU packages not installed")
        print("   ℹ Install with: pip install torch gpytorch")
        gpu_success = False
        gpu_time = None
    except Exception as e:
        print(f"   ✗ GPU training failed: {e}")
        gpu_success = False
        gpu_time = None

    # Comparison
    print("\n3. Performance Comparison")
    print("   " + "-" * 40)
    if cpu_success and gpu_success:
        speedup = cpu_time / gpu_time
        print(f"   CPU time:    {format_time(cpu_time)}")
        print(f"   GPU time:    {format_time(gpu_time)}")
        print(f"   Speedup:     {speedup:.1f}x")
        print(f"   Time saved:  {format_time(cpu_time - gpu_time)}")

        if speedup > 5:
            print(f"GPU is significantly faster!")
        elif speedup > 2:
            print(f"GPU provides good speedup")
        else:
            print(f"Limited speedup (small dataset)")

        return cpu_time, gpu_time, speedup
    elif cpu_success:
        print(f"   CPU time: {format_time(cpu_time)}")
        print(f"   GPU test skipped")
        return cpu_time, None, None
    else:
        print(f"   ✗ Both tests failed")
        return None, None, None


def compare_predictions():
    """Compare prediction accuracy between CPU and GPU."""
    print("\n" + "=" * 70)
    print("PREDICTION COMPARISON")
    print("=" * 70)

    # Prepare small dataset
    from multifidelity_gpr import MultiFidelityGPR
    model_temp = MultiFidelityGPR()
    x_grid, y_grid, data = model_temp.load_data('truncated_data')
    X_hf, y_hf, y_lf = model_temp.prepare_training_data(data)

    n_train = 2000
    indices = np.random.choice(len(y_hf), n_train, replace=False)
    X_train = X_hf[indices]
    y_train = y_hf[indices]
    y_lf_train = y_lf[indices]

    # Train both models
    print("\nTraining models...")
    model_cpu = MultiFidelityGPR()
    model_cpu.load_data('truncated_data')
    model_cpu.fit(X_train, y_train, y_lf_train)
    print("✓ CPU model trained")

    try:
        import torch
        if torch.cuda.is_available():
            from multifidelity_gpr_gpu import MultiFidelityGPR_GPU

            model_gpu = MultiFidelityGPR_GPU(device='cuda')
            model_gpu.load_data('truncated_data')
            model_gpu.fit(X_train, y_train, y_lf_train, training_iter=50)
            print("✓ GPU model trained")

            # Compare predictions
            print("\nComparing predictions at different angles...")
            test_angles = [0, 45, 90, 180, 270]

            print(f"\n{'Angle':<10} {'Mean Diff':<15} {'Max Diff':<15} {'Rel. Diff %':<15}")
            print("-" * 55)

            for angle in test_angles:
                pred_cpu, _ = model_cpu.predict_field(angle, data)
                pred_gpu, _ = model_gpu.predict_field(angle, data)

                diff = np.abs(pred_cpu - pred_gpu)
                mean_diff = diff.mean()
                max_diff = diff.max()
                rel_diff = mean_diff / pred_cpu.mean() * 100

                print(f"{angle:<10} {mean_diff:<15.6f} {max_diff:<15.6f} {rel_diff:<15.3f}")

            print("\nℹ Small differences are expected due to:")
            print("  • Different optimization algorithms (L-BFGS-B vs Adam)")
            print("  • Floating-point precision differences")
            print("  • Random initialization")

        else:
            print("✗ GPU not available - skipping prediction comparison")

    except ImportError:
        print("✗ GPU packages not installed - skipping prediction comparison")


def create_speedup_plot():
    """Create a speedup visualization for different dataset sizes."""
    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)

    try:
        import torch
        if not torch.cuda.is_available():
            print("GPU not available - skipping scalability analysis")
            return

        from multifidelity_gpr import MultiFidelityGPR
        from multifidelity_gpr_gpu import MultiFidelityGPR_GPU

        # Load data
        print("\nLoading data...")
        model_temp = MultiFidelityGPR()
        x_grid, y_grid, data = model_temp.load_data('truncated_data')
        X_hf, y_hf, y_lf = model_temp.prepare_training_data(data)

        # Test different sizes
        sizes = [500, 1000, 2000, 5000, 10000]
        cpu_times = []
        gpu_times = []
        speedups = []

        print(f"\n{'Size':<10} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
        print("-" * 45)

        for size in sizes:
            if size > len(y_hf):
                break

            indices = np.random.choice(len(y_hf), size, replace=False)
            X = X_hf[indices]
            y = y_hf[indices]
            ylf = y_lf[indices]

            # CPU timing
            model_cpu = MultiFidelityGPR()
            model_cpu.load_data('truncated_data')
            start = time.time()
            model_cpu.fit(X, y, ylf)
            cpu_time = time.time() - start
            cpu_times.append(cpu_time)

            # GPU timing
            model_gpu = MultiFidelityGPR_GPU(device='cuda')
            model_gpu.load_data('truncated_data')
            start = time.time()
            model_gpu.fit(X, y, ylf, training_iter=50)
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)

            speedup = cpu_time / gpu_time
            speedups.append(speedup)

            print(f"{size:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.1f}x")

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time comparison
        ax1.plot(sizes[:len(cpu_times)], cpu_times, 'o-', label='CPU', linewidth=2, markersize=8)
        ax1.plot(sizes[:len(gpu_times)], gpu_times, 's-', label='GPU', linewidth=2, markersize=8)
        ax1.set_xlabel('Dataset Size', fontsize=12)
        ax1.set_ylabel('Training Time (seconds)', fontsize=12)
        ax1.set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Speedup
        ax2.plot(sizes[:len(speedups)], speedups, 'o-', color='green', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_xlabel('Dataset Size', fontsize=12)
        ax2.set_ylabel('Speedup Factor', fontsize=12)
        ax2.set_title('GPU Speedup vs Dataset Size', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

        plt.tight_layout()
        plt.savefig('cpu_gpu_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot: cpu_gpu_comparison.png")

    except ImportError:
        print("✗ GPU packages not installed")
    except Exception as e:
        print(f"✗ Scalability analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare CPU and GPU performance for multi-fidelity GPR"
    )
    parser.add_argument('--size', type=int, default=5000,
                       help='Number of training samples (default: 5000)')
    parser.add_argument('--skip-scalability', action='store_true',
                       help='Skip scalability analysis')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 15 + "CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 70 + "\n")

    # Training comparison
    cpu_time, gpu_time, speedup = compare_training(n_samples=args.size)

    # Prediction comparison
    compare_predictions()

    # Scalability analysis
    if not args.skip_scalability:
        create_speedup_plot()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if speedup:
        print(f"Training speedup: {speedup:.1f}x faster on GPU")
        print(f"Time saved: {format_time(cpu_time - gpu_time)}")

        if speedup > 10:
            print("\nExcellent! GPU provides significant acceleration.")
            print("   Recommended for production use with large datasets.")
        elif speedup > 5:
            print("\nGood speedup! GPU recommended for faster iterations.")
        elif speedup > 2:
            print("\nModerate speedup. GPU useful for larger datasets.")
        else:
            print("\nLimited speedup with current dataset size.")
            print("GPU benefits increase with larger datasets (>10k points).")
    else:
        print("GPU comparison not available")
        print("  Install GPU support: pip install torch gpytorch")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

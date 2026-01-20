import argparse
import itertools
import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


# Hyperparameter ranges for basic MultiFidelityGPR model
BASIC_HYPERPARAMETERS = {
    'kernel_length_scale': [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    'noise_level': [0.001, 0.005, 0.01, 0.05, 0.1],
    'training_iter': [200],
    'subsample_ratio': [0.1, 0.2, 0.3, 0.5],
    'batch_size': [500,1000,5000,10000], 
}

# Hyperparameter ranges for advanced MultiFidelityGPR model
ADVANCED_HYPERPARAMETERS = {
    'spatial_length_scale': [20.0, 50.0, 100.0, 150.0],
    'angular_length_scale': [5.0, 10.0, 15.0, 20.0],
    'use_matern': [True, False],
    'training_iter': [30, 50, 100],
    'subsample_spatial': [1, 2, 3],
}


QUICK_BASIC_HYPERPARAMETERS = {
    'kernel_length_scale': [10.0, 20.0],
    'noise_level': [0.01, 0.05],
    'training_iter': [30],
    'subsample_ratio': [0.2],
    'batch_size': [10000],
}

QUICK_ADVANCED_HYPERPARAMETERS = {
    'spatial_length_scale': [50.0, 100.0],
    'angular_length_scale': [10.0, 15.0],
    'use_matern': [True],
    'training_iter': [30],
    'subsample_spatial': [2],
}


def generate_model_name(model_type, params, timestamp=None):
    """
        Generate a descriptive model filename encoding all hyperparameters.

        Example: basic_kls15.0_noise0.01_iter50_sub0.2_bs10000_20260116_143022.pth
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [model_type]

    # Sort parameters for consistent naming
    for key in sorted(params.keys()):
        value = params[key]
        # Create short abbreviations
        abbrev = {
            'kernel_length_scale': 'kls',
            'noise_level': 'noise',
            'training_iter': 'iter',
            'subsample_ratio': 'sub',
            'batch_size': 'bs',
            'spatial_length_scale': 'sls',
            'angular_length_scale': 'als',
            'use_matern': 'matern',
            'subsample_spatial': 'sspt',
        }.get(key, key)

        # Format value
        if isinstance(value, bool):
            value_str = 'T' if value else 'F'
        elif isinstance(value, float):
            value_str = f"{value:.3f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)

        parts.append(f"{abbrev}{value_str}")

    parts.append(timestamp)

    return '_'.join(parts)


def train_and_evaluate_basic(params, use_gpu=True, device='cuda', output_dir='sweep_results'):
    """
    Train and evaluate a basic MultiFidelityGPR model with given hyperparameters.

    Returns:
        dict: Results including metrics and model path
    """
    print(f"\n{'='*80}")
    print(f"Training Basic Model with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print('='*80)

    try:
        start_time = time.time()

        # Import model
        if use_gpu:
            from multifidelity_gpr_gpu import MultiFidelityGPR_GPU
            model = MultiFidelityGPR_GPU(
                kernel_length_scale=params['kernel_length_scale'],
                noise_level=params['noise_level'],
                device=device,
                max_training_samples=None  # Use batch training
            )
        else:
            from multifidelity_gpr import MultiFidelityGPR
            model = MultiFidelityGPR(
                kernel_length_scale=params['kernel_length_scale'],
                noise_level=params['noise_level']
            )

        # Load data
        print("Loading data...")
        x_grid, y_grid, data = model.load_data('truncated_data')

        # Prepare training data
        print("Preparing training data...")
        X_hf, y_hf, y_lf = model.prepare_training_data(data)

        # Subsample
        n_sample = int(len(y_hf) * params['subsample_ratio'])
        print(f"Using {n_sample:,} of {len(y_hf):,} points ({params['subsample_ratio']*100:.0f}%)")
        indices = np.random.choice(len(y_hf), n_sample, replace=False)
        X_train = X_hf[indices]
        y_train = y_hf[indices]
        y_lf_train = y_lf[indices]

        # Train model
        print(f"Training model (iterations: {params['training_iter']})...")
        if use_gpu:
            model.fit(X_train, y_train, y_lf_train,
                     training_iter=params['training_iter'],
                     use_batch_training=True,
                     batch_size=params['batch_size'])
        else:
            model.fit(X_train, y_train, y_lf_train)

        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.1f} seconds")

        # Evaluate on test angles
        print("Evaluating on test set...")
        les_angles = data['les']['angles']
        les_data_arr = data['les']['data']

        # Use a subset for evaluation to save time
        test_indices = list(range(0, len(les_angles), max(1, len(les_angles)//10)))

        rmse_scores = []
        mae_scores = []
        r2_scores = []

        for i in tqdm(test_indices, desc="Evaluating test angles", unit="angle"):
            angle = les_angles[i]
            pred, _ = model.predict_field(angle, data)
            les_field = les_data_arr[i]
            mask = les_field > 1e-6

            if mask.sum() > 0:
                # RMSE
                rmse = np.sqrt(np.mean((pred[mask] - les_field[mask])**2))
                rmse_scores.append(rmse)

                # MAE
                mae = np.mean(np.abs(pred[mask] - les_field[mask]))
                mae_scores.append(mae)

                # R²
                ss_res = np.sum((les_field[mask] - pred[mask])**2)
                ss_tot = np.sum((les_field[mask] - np.mean(les_field[mask]))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r2_scores.append(r2)

        # Compute aggregate metrics
        mean_rmse = np.mean(rmse_scores)
        mean_mae = np.mean(mae_scores)
        mean_r2 = np.mean(r2_scores)

        print(f"\nEvaluation Results:")
        print(f"  Mean RMSE: {mean_rmse:.5f}")
        print(f"  Mean MAE:  {mean_mae:.5f}")
        print(f"  Mean R²:   {mean_r2:.5f}")

        # Generate model name and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = generate_model_name('basic', params, timestamp)
        model_path = os.path.join(output_dir, f"{model_name}.pth" if use_gpu else f"{model_name}.pkl")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving model to {model_path}...")
        model.save_model(model_path)

        # Save metadata
        metadata = {
            'model_type': 'basic',
            'hyperparameters': params,
            'metrics': {
                'mean_rmse': float(mean_rmse),
                'mean_mae': float(mean_mae),
                'mean_r2': float(mean_r2),
                'rmse_std': float(np.std(rmse_scores)),
                'mae_std': float(np.std(mae_scores)),
                'r2_std': float(np.std(r2_scores)),
            },
            'training_time_seconds': training_time,
            'n_training_samples': n_sample,
            'timestamp': timestamp,
            'model_path': model_path,
        }

        metadata_path = model_path.replace('.pth', '.json').replace('.pkl', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_path}")
        print("Training and evaluation complete...\n")

        return metadata

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_and_evaluate_advanced(params, use_gpu=True, device='cuda', output_dir='sweep_results'):
    """
        Train and evaluate an advanced MultiFidelityGPR model with given hyperparameters.

        Returns:
            dict: Results including metrics and model path
    """
    print(f"\n{'='*80}")
    print(f"Training Advanced Model with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print('='*80)

    try:
        start_time = time.time()

        # Import model
        if use_gpu:
            from advanced_multifidelity_gpr_gpu import AdvancedMultiFidelityGPR_GPU
            model = AdvancedMultiFidelityGPR_GPU(
                spatial_length_scale=params['spatial_length_scale'],
                angular_length_scale=params['angular_length_scale'],
                use_matern=params['use_matern'],
                device=device
            )
        else:
            from advanced_multifidelity_gpr import AdvancedMultiFidelityGPR
            model = AdvancedMultiFidelityGPR(
                spatial_length_scale=params['spatial_length_scale'],
                angular_length_scale=params['angular_length_scale'],
                use_matern=params['use_matern']
            )

        # Load data
        print("Loading data...")
        data = model.load_data('truncated_data')

        # Prepare training data
        print("Preparing training data...")
        X_hf, y_hf, y_lf = model.prepare_training_data(
            subsample_spatial=params['subsample_spatial']
        )

        # Train model
        print(f"Training model (iterations: {params['training_iter']})...")
        model.fit(X_hf, y_hf, y_lf, method='autoregressive',
                 training_iter=params['training_iter'])

        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.1f} seconds")

        # Evaluate
        print("Evaluating on test set...")
        results = model.evaluate_predictions(n_test_angles=10)

        print(f"\nEvaluation Results:")
        print(f"  Mean RMSE: {results['mean_rmse']:.5f}")
        print(f"  Mean MAE:  {results['mean_mae']:.5f}")
        print(f"  Mean R²:   {results['mean_r2']:.5f}")

        # Generate model name and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = generate_model_name('advanced', params, timestamp)
        model_path = os.path.join(output_dir, f"{model_name}.pth" if use_gpu else f"{model_name}.pkl")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving model to {model_path}...")
        model.save_model(model_path)

        # Save metadata
        metadata = {
            'model_type': 'advanced',
            'hyperparameters': params,
            'metrics': results,
            'training_time_seconds': training_time,
            'n_training_samples': len(y_hf),
            'timestamp': timestamp,
            'model_path': model_path,
        }

        metadata_path = model_path.replace('.pth', '.json').replace('.pkl', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_path}")
        print("✓ Training and evaluation complete\n")

        return metadata

    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_hyperparameter_sweep(model_type='basic', use_gpu=True, device='cuda',
                             output_dir='sweep_results', quick_test=False,
                             parallel=False, n_jobs=-1):
    """
        Run a complete hyperparameter sweep.

        Args:
            model_type: 'basic' or 'advanced'
            use_gpu: Use GPU acceleration
            device: Device to use
            output_dir: Output directory for models and results
            quick_test: Run quick test with fewer combinations
            parallel: Run in parallel (requires joblib)
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            pd.DataFrame: Summary of all experiments
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP")
    print(f"Model Type: {model_type}")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")
    print(f"Quick Test: {quick_test}")
    print(f"Parallel: {parallel}")
    print("="*80 + "\n")

    # Select hyperparameter grid
    if model_type == 'basic':
        param_grid = QUICK_BASIC_HYPERPARAMETERS if quick_test else BASIC_HYPERPARAMETERS
        train_fn = train_and_evaluate_basic
    elif model_type == 'advanced':
        param_grid = QUICK_ADVANCED_HYPERPARAMETERS if quick_test else ADVANCED_HYPERPARAMETERS
        train_fn = train_and_evaluate_advanced
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Total combinations to test: {len(combinations)}")
    print(f"Hyperparameters: {param_names}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run sweep
    all_results = []

    if parallel:
        try:
            from joblib import Parallel, delayed
            print(f"Running in parallel with {n_jobs} jobs...\n")

            def train_wrapper(combo):
                params = dict(zip(param_names, combo))
                return train_fn(params, use_gpu=use_gpu, device=device, output_dir=output_dir)

            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(train_wrapper)(combo) for combo in combinations
            )
            all_results = [r for r in results if r is not None]

        except ImportError:
            print("WARNING: joblib not installed. Running sequentially.")
            print("Install with: pip install joblib\n")
            parallel = False

    if not parallel:
        print("Running sequentially...\n")
        for i, combo in enumerate(tqdm(combinations, desc="Hyperparameter sweep", unit="config"), 1):
            print(f"\n>>> Experiment {i}/{len(combinations)}")
            params = dict(zip(param_names, combo))
            result = train_fn(params, use_gpu=use_gpu, device=device, output_dir=output_dir)
            if result is not None:
                all_results.append(result)

    # Create summary DataFrame
    if all_results:
        summary_data = []
        for result in all_results:
            row = {
                'model_path': result['model_path'],
                'timestamp': result['timestamp'],
                'training_time': result['training_time_seconds'],
                **result['hyperparameters'],
                **{f'metric_{k}': v for k, v in result['metrics'].items()},
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Sort by RMSE (lower is better)
        df = df.sort_values('metric_mean_rmse')

        # Save summary
        summary_path = os.path.join(output_dir, f'sweep_summary_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"Sweep complete! Summary saved to: {summary_path}")
        print(f"Total successful experiments: {len(all_results)}/{len(combinations)}")
        print(f"\nTop 5 models by RMSE:")
        print("="*80)

        # Display top models
        display_cols = ['model_path', 'metric_mean_rmse', 'metric_mean_mae', 'metric_mean_r2']
        display_cols.extend([c for c in df.columns if c in param_names])
        print(df[display_cols].head(5).to_string(index=False))

        print("\n" + "="*80)
        print(f"\nBest model: {df.iloc[0]['model_path']}")
        print("="*80 + "\n")

        return df
    else:
        print("\n✗ No successful experiments completed")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for Multi-Fidelity GPR models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model-type', type=str, default='basic',
                       choices=['basic', 'advanced'],
                       help='Which model to sweep (default: basic)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                       help='Output directory for models (default: sweep_results)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with subset of combinations')
    parser.add_argument('--parallel', action='store_true',
                       help='Run sweep in parallel (requires joblib)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (default: -1 = all cores)')

    args = parser.parse_args()

    # Check GPU availability
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
                print(f"\n  GPU Acceleration Enabled: {device}")
                if device == 'cuda':
                    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("WARNING: PyTorch not installed. GPU acceleration disabled.")
            print("Install with: pip install torch gpytorch")
            args.use_gpu = False
            device = 'cpu'
    else:
        device = 'cpu'

    # Run sweep
    start_time = time.time()
    df = run_hyperparameter_sweep(
        model_type=args.model_type,
        use_gpu=args.use_gpu,
        device=device,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
        parallel=args.parallel,
        n_jobs=args.n_jobs
    )
    total_time = time.time() - start_time

    print(f"\nTotal sweep time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Models and results saved in: {args.output_dir}/\n")


if __name__ == "__main__":
    main()

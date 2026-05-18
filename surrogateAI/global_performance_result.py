import os
import pickle
import numpy as np
import argparse
from sklearn.metrics import r2_score

def load_rollout(file_path):
    """Loads the pickle file containing rollout data."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Successfully loaded {len(data)} samples from {os.path.basename(file_path)}")
    return data

def calculate_global_metrics(pkl_file_path, output_dir):
    """
    Reads the rollout.pkl, calculates metrics per sample, 
    and saves the global Mean ± Std Dev to a text file.
    """
    data = load_rollout(pkl_file_path)
    if not data:
        return

    # Lists to store the individual metric scores for each of the 40 samples
    sample_r2s = []
    sample_rmses = []
    sample_mapes = []
    sample_max_errors = []

    print("\nProcessing samples...")
    for i, sample in enumerate(data):
        # Extract the flattened prediction and target arrays
        target = sample['target'].flatten()
        prediction = sample['prediction'].flatten()
        
        # 1. R2 Score
        if len(target) > 1 and np.var(target) > 1e-9:
            r2 = r2_score(target, prediction)
        else:
            r2 = 0.0
        sample_r2s.append(r2)
        
        # 2. RMSE (Log Space)
        mse = np.mean((target - prediction) ** 2)
        rmse = np.sqrt(mse)
        sample_rmses.append(rmse)
        
        # 3. MAPE (Real Space)
        real_target = np.power(10, target)
        real_pred = np.power(10, prediction)
        # Avoid division by zero
        ape = np.abs((real_target - real_pred) / (real_target + 1e-9)) * 100.0
        mape = np.mean(ape)
        sample_mapes.append(mape)
        
        # 4. Max Absolute Error (Log Space)
        max_err = np.max(np.abs(target - prediction))
        sample_max_errors.append(max_err)

    # Calculate Global Statistics (Mean and Standard Deviation)
    mean_r2 = np.mean(sample_r2s)
    std_r2 = np.std(sample_r2s)
    
    mean_rmse = np.mean(sample_rmses)
    std_rmse = np.std(sample_rmses)
    
    mean_mape = np.mean(sample_mapes)
    std_mape = np.std(sample_mapes)
    
    mean_max_err = np.mean(sample_max_errors)
    std_max_err = np.std(sample_max_errors)

    # Generate the formatted report string
    report = (
        f"======================================================\n"
        f"FINAL GLOBAL METRICS (MEAN ± STD DEV)\n"
        f"======================================================\n"
        f"R² Score        : {mean_r2:.3f} ± {std_r2:.3f}\n"
        f"RMSE (Log Space): {mean_rmse:.3f} ± {std_rmse:.3f}\n"
        f"MAPE (%)        : {mean_mape:.1f}% ± {std_mape:.1f}%\n"
        f"Max Error (Log) : {mean_max_err:.3f} ± {std_max_err:.3f}\n"
        f"======================================================\n"
    )

    # Print to console
    print(report)
    print("Copy these values directly into your Master Benchmarking Table!\n")

    # Ensure output directory exists and save to TXT
    os.makedirs(output_dir, exist_ok=True)
    out_file_path = os.path.join(output_dir, "global_benchmarking_metrics.txt")
    
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Saved global metrics to: {out_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate global benchmark metrics from rollout pkl.")
    
    # Target PKL file
    parser.add_argument('--file', type=str, 
                        default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/full_regressor/regDGCNN_seg/step_shaft_tensile_torsion_enriched9/EXP_full_relative_loss_onecycle_k50_step_tensile_torsion_enriched9/2026-03-16_13-48-45/rollout/rollout_epoch_610.pkl")
    
    # Output directory for the TXT file
    parser.add_argument('--output_dir', type=str, 
                        default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/full_regressor/regDGCNN_seg/step_shaft_tensile_torsion_enriched9/EXP_full_relative_loss_onecycle_k50_step_tensile_torsion_enriched9/2026-03-16_13-48-45/global_performance_plot_610")
    
    args = parser.parse_args()
    
    calculate_global_metrics(args.file, args.output_dir)
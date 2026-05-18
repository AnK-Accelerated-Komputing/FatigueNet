import matplotlib
# Force matplotlib to not use any Xwindows backend (Server safe)
matplotlib.use('Agg') 

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
import argparse
import re
from sklearn.metrics import r2_score

def load_rollout(file_path):
    """Loads the pickle file containing rollout data."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples from {file_path}")
    return data

def extract_epoch_from_filename(filepath):
    """
    Extracts epoch number from filename like 'rollout_epoch_1000.pkl'
    """
    filename = os.path.basename(filepath)
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"

def calculate_and_save_metrics(mesh_pos, target, prediction, group_id, sample_index, output_dir):
    """
    Calculates detailed metrics and saves them to CSV and TXT files.
    """
    # 1. Calculate Per-Node Errors (Log Scale)
    abs_error_log = np.abs(target - prediction)
    
    # 2. Calculate Real Life Cycles (Inverse Log10)
    real_life_gt = np.power(10, target)
    real_life_pred = np.power(10, prediction)
    
    # 3. Calculate Errors on Real Scale (Non-Log)
    real_abs_error = np.abs(real_life_gt - real_life_pred)
    
    # APE (Avoid division by zero)
    ape = np.abs((real_life_gt - real_life_pred) / (real_life_gt + 1e-9)) * 100.0

    # 4. Create DataFrame for CSV
    df = pd.DataFrame({
        'Node_Index': np.arange(len(target)),
        'Mesh_X': mesh_pos[:, 0],
        'Mesh_Y': mesh_pos[:, 1],
        'Mesh_Z': mesh_pos[:, 2],
        'Log_Life_GT': target,
        'Log_Life_Pred': prediction,
        'Life_GT': real_life_gt,      # Actual Cycles
        'Life_Pred': real_life_pred,  # Actual Cycles
        'Abs_Error_Log': abs_error_log, # Error in Log Scale
        'Abs_Error_Real': real_abs_error, # Error in Actual Cycles
        'Abs_Percentage_Error': ape
    })
    
    # Save CSV
    safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")
    csv_filename = f"analysis_idx{sample_index}_{safe_group_id}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"   Saved detailed CSV to: {csv_path}")

    # 5. Calculate Summary Metrics
    mse = np.mean((target - prediction) ** 2)
    rmse = np.sqrt(mse)
    mae_log = np.mean(abs_error_log)
    mae_real = np.mean(real_abs_error)
    mape = np.mean(ape)
    ape_max = np.max(ape)
    
    # R2 Score (Handle case with single value or constant target)
    if len(target) > 1 and np.var(target) > 1e-9:
        r2 = r2_score(target, prediction)
    else:
        r2 = 0.0
        
    max_gt = np.max(target)
    max_gt_real = np.max(real_life_gt)
    max_pred_real = np.max(real_life_pred)
    max_pred = np.max(prediction)
    max_error_log = np.max(abs_error_log)
    max_error_real = np.max(real_abs_error)

    # 6. Generate Summary Report Text
    report = (
        f"--- Metrics Report for Sample {sample_index} ---\n"
        f"Group ID: {group_id}\n"
        f"------------------------------------------\n"
        f"Total Nodes: {len(target)}\n\n"
        f"1. Global Error Metrics:\n"
        f"   R2 Score         : {r2:.6f}\n"
        f"   RMSE (Log Space)  : {rmse:.6f}\n"
        f"   MSE (Log Space)  : {mse:.6f}\n"
        f"   MAE (Log Space)  : {mae_log:.6f}\n"
        f"   MAE (Real Cycles): {mae_real:.2f}\n"
        f"   MAPE (Mean %)    : {mape:.4f}%\n\n"
        f"   Max APE           : {ape_max:.2f}%\n\n"
        f"2. Peak Values:\n"
        f"   Max Life (Log)   : {max_gt:.6f}\n"
        f"   Max Pred (Log)   : {max_pred:.6f}\n"
        f"   Max Error (Log)  : {max_error_log:.6f}\n"
        f"   Max Error (Real) : {max_error_real:.2f} cycles\n"
        f"   Max Life (Real)  : {max_gt_real:.2f} cycles\n"
        f"   Max Pred (Real)  : {max_pred_real:.2f} cycles\n"
    )

    # Save Text Report
    txt_filename = f"summary_idx{sample_index}_{safe_group_id}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w") as f:
        f.write(report)
    
    print(f"   Saved summary stats to: {txt_path}")
    
    # Return metrics for display on plot title
    return r2, mse, max_error_log

def visualize_rollout(rollout_data, sample_index, output_dir):
    """
    Visualizes a single sample, SAVES image, SAVES CSV, and SAVES Summary TXT.
    """
    if rollout_data is None or len(rollout_data) == 0:
        print("No data to visualize.")
        return

    if sample_index >= len(rollout_data):
        print(f"Error: Index {sample_index} out of range (Max: {len(rollout_data)-1})")
        return

    # Get sample
    sample = rollout_data[sample_index]
    
    # Extract data
    mesh_pos = sample['mesh_pos']
    if hasattr(mesh_pos, 'cpu'): mesh_pos = mesh_pos.cpu().numpy()
    
    target = sample['target'].flatten()
    prediction = sample['prediction'].flatten()
    group_id = sample.get('group_id', 'Unknown')
    
    print(f"Processing Sample {sample_index} (Group: {group_id})...")
    
    # --- STEP 1: CALCULATE & SAVE METRICS (CSV/TXT) ---
    r2, mse, max_err = calculate_and_save_metrics(
        mesh_pos, target, prediction, group_id, sample_index, output_dir
    )

    # --- STEP 2: PLOTTING ---
    fig = plt.figure(figsize=(20, 6))
    
    # Common color scale
    vmin, vmax = target.min(), target.max()
    
    # Plot 1: Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    p1 = ax1.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
                     c=target, cmap='jet', s=3, vmin=vmin, vmax=vmax)
    ax1.set_title(f'Ground Truth (Log Life)\nMax Life: {target.max():.2f}')
    fig.colorbar(p1, ax=ax1, shrink=0.6)

    # Plot 2: Prediction
    ax2 = fig.add_subplot(132, projection='3d')
    p2 = ax2.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
                     c=prediction, cmap='jet', s=3, vmin=vmin, vmax=vmax)
    ax2.set_title(f'Prediction\nR2: {r2:.4f} | MSE: {mse:.4f}')
    fig.colorbar(p2, ax=ax2, shrink=0.6)

    # Plot 3: Absolute Error (Log Scale)
    error = np.abs(target - prediction)
    ax3 = fig.add_subplot(133, projection='3d')
    p3 = ax3.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
                     c=error, cmap='inferno', s=3)
    ax3.set_title(f'Absolute Error (Log Scale)\nMax Error: {max_err:.4f}')
    fig.colorbar(p3, ax=ax3, shrink=0.6, label='Error Magnitude')

    # Save Image
    safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")
    filename = f"plot_idx{sample_index}_{safe_group_id}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    print(f"   Saved visualization plot to: {save_path}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze rollout: CSV, Metrics, and Plots.")
    parser.add_argument('--file', type=str, default="/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/regDGCNN_seg/fatigue_dataset/EXP_full_run_k50_drop_0.1_dim_2048/2026-02-12_05-50-41/rollout/rollout_epoch_1660.pkl", help="Path to the rollout .pkl file")
    
    # 3 Ways to select data
    parser.add_argument('--index', type=int, default=None, help="Process a specific index.")
    parser.add_argument('--group_id', type=str, default="group_86", help="Process a specific group (e.g., 'group_86')")
    parser.add_argument('--all', action='store_true', help="Process ALL samples in the rollout file.")
    
    parser.add_argument('--output_dir', type=str, default="/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/regDGCNN_seg/fatigue_dataset/EXP_full_run_k50_drop_0.1_dim_2048/2026-02-12_05-50-41/rollout_csv", help="Base directory for outputs")
    
    args = parser.parse_args()
    
    # 1. Extract Epoch Number
    epoch_str = extract_epoch_from_filename(args.file)
    
    # 2. Create Sub-directory based on Epoch
    final_output_dir = os.path.join(args.output_dir, f"rollout_epoch_{epoch_str}")
    
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        print(f"Created output directory: {final_output_dir}")
    else:
        print(f"Saving to existing directory: {final_output_dir}")

    # 3. Load Data
    data = load_rollout(args.file)
    if data is None:
        exit()

    # --- NEW SELECTION LOGIC ---
    if args.all:
        print(f"Processing ALL {len(data)} samples...")
        for idx in range(len(data)):
            visualize_rollout(data, sample_index=idx, output_dir=final_output_dir)
            
    elif args.group_id is not None:
        print(f"Searching for {args.group_id}...")
        found_idx = -1
        available_groups = []
        
        # Scan the rollout to find the matching group
        for idx, sample in enumerate(data):
            g_id = str(sample.get('group_id', 'Unknown'))
            available_groups.append(g_id)
            if g_id == args.group_id:
                found_idx = idx
                break
                
        if found_idx != -1:
            visualize_rollout(data, sample_index=found_idx, output_dir=final_output_dir)
        else:
            print(f"\n[ERROR] Group ID '{args.group_id}' was NOT found in this rollout.")
            print(f"Here are the groups available in this file (showing first 30):")
            print(f"{available_groups[:30]}...")
            
    else:
        # Default fallback to index (defaults to 0 if neither is provided)
        target_idx = args.index if args.index is not None else 0
        visualize_rollout(data, sample_index=target_idx, output_dir=final_output_dir)
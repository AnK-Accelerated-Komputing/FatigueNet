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

def calculate_and_save_metrics(mesh_pos, target, prediction, group_id, sample_index, output_dir):
    """
    Calculates detailed metrics and saves them to CSV and TXT files.
    """
    # 1. Calculate Per-Node Errors
    abs_error = np.abs(target - prediction)
    # APE (Avoid division by zero)
    ape = np.abs((target - prediction) / (target + 1e-9)) * 100.0
    
    # 2. Create DataFrame for CSV
    df = pd.DataFrame({
        'Node_Index': np.arange(len(target)),
        'Mesh_X': mesh_pos[:, 0],
        'Mesh_Y': mesh_pos[:, 1],
        'Mesh_Z': mesh_pos[:, 2],
        'Log_Life_GT': target,
        'Log_Life_Pred': prediction,
        'Abs_Error': abs_error,
        'Abs_Percentage_Error': ape
    })
    
    # Save CSV
    safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")
    csv_filename = f"analysis_idx{sample_index}_{safe_group_id}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"   Saved detailed CSV to: {csv_path}")

    # 3. Calculate Summary Metrics
    mse = np.mean((target - prediction) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    mape = np.mean(ape)
    
    # R2 Score (Handle case with single value or constant target)
    if len(target) > 1 and np.var(target) > 1e-9:
        r2 = r2_score(target, prediction)
    else:
        r2 = 0.0
        
    max_gt = np.max(target)
    max_pred = np.max(prediction)
    max_error = np.max(abs_error)
    min_gt = np.min(target)

    # 4. Generate Summary Report Text
    report = (
        f"--- Metrics Report for Sample {sample_index} ---\n"
        f"Group ID: {group_id}\n"
        f"------------------------------------------\n"
        f"Total Nodes: {len(target)}\n\n"
        f"1. Global Error Metrics:\n"
        f"   R2 Score         : {r2:.6f}\n"
        f"   MSE (Mean Sq)    : {mse:.6f}\n"
        f"   RMSE (Root MSE)  : {rmse:.6f}\n"
        f"   MAE (Mean Abs)   : {mae:.6f}\n"
        f"   MAPE (Mean %)    : {mape:.4f}%\n\n"
        f"2. Peak Values (Log Scale):\n"
        f"   Max Ground Truth : {max_gt:.6f}\n"
        f"   Max Prediction   : {max_pred:.6f}\n"
        f"   Min Ground Truth : {min_gt:.6f}\n"
        f"   Max Abs Error    : {max_error:.6f}\n"
    )

    # Save Text Report
    txt_filename = f"summary_idx{sample_index}_{safe_group_id}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w") as f:
        f.write(report)
    
    print(f"   Saved summary stats to: {txt_path}")
    
    # Return metrics for display on plot title
    return r2, mse, max_error

def visualize_rollout(rollout_data, sample_index=0, output_dir="."):
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

    # Plot 3: Absolute Error
    error = np.abs(target - prediction)
    ax3 = fig.add_subplot(133, projection='3d')
    p3 = ax3.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
                     c=error, cmap='inferno', s=3)
    ax3.set_title(f'Absolute Error\nMax Error: {max_err:.4f}')
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
    parser.add_argument('--file', type=str, default="/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/regDGCNN_seg/fatigue_dataset/EXP_full_run_hybrid_K60_drop_0.1_lr2e-4_cosine/2026-02-16_01-23-43/rollout/rollout_epoch_750.pkl")
    parser.add_argument('--index', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/regDGCNN_seg/fatigue_dataset/EXP_full_run_hybrid_K60_drop_0.1_lr2e-4_cosine/2026-02-16_01-23-43/rollout_csv")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = load_rollout(args.file)
    visualize_rollout(data, sample_index=args.index, output_dir=args.output_dir)
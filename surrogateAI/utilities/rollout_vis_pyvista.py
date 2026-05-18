import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import re
from sklearn.metrics import r2_score
import pyvista as pv

# ---- Important for headless servers ----
pv.OFF_SCREEN = True
os.environ["PYVISTA_OFF_SCREEN"] = "true"


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
    filename = os.path.basename(filepath)
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"


def calculate_and_save_metrics(mesh_pos, target, prediction, group_id, sample_index, output_dir):

    abs_error_log = np.abs(target - prediction)

    real_life_gt = np.power(10, target)
    real_life_pred = np.power(10, prediction)

    real_abs_error = np.abs(real_life_gt - real_life_pred)
    ape = np.abs((real_life_gt - real_life_pred) / (real_life_gt + 1e-9)) * 100.0

    df = pd.DataFrame({
        'Node_Index': np.arange(len(target)),
        'Mesh_X': mesh_pos[:, 0],
        'Mesh_Y': mesh_pos[:, 1],
        'Mesh_Z': mesh_pos[:, 2],
        'Log_Life_GT': target,
        'Log_Life_Pred': prediction,
        'Life_GT': real_life_gt,
        'Life_Pred': real_life_pred,
        'Abs_Error_Log': abs_error_log,
        'Abs_Error_Real': real_abs_error,
        'Abs_Percentage_Error': ape
    })

    safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")

    csv_filename = f"analysis_idx{sample_index}_{safe_group_id}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    df.to_csv(csv_path, index=False)

    print(f"   Saved detailed CSV to: {csv_path}")

    mse = np.mean((target - prediction) ** 2)
    rmse = np.sqrt(mse)

    mae_log = np.mean(abs_error_log)
    mae_real = np.mean(real_abs_error)
    mape = np.mean(ape)
    ape_max = np.max(ape)

    if len(target) > 1 and np.var(target) > 1e-9:
        r2 = r2_score(target, prediction)
    else:
        r2 = 0.0

    max_gt = np.max(target)
    max_pred = np.max(prediction)

    max_gt_real = np.max(real_life_gt)
    max_pred_real = np.max(real_life_pred)

    max_error_log = np.max(abs_error_log)
    max_error_real = np.max(real_abs_error)

    report = (
        f"--- Metrics Report for Sample {sample_index} ---\n"
        f"Group ID: {group_id}\n"
        f"------------------------------------------\n"
        f"Total Nodes: {len(target)}\n\n"
        f"1. Global Error Metrics:\n"
        f"   R2 Score         : {r2:.6f}\n"
        f"   RMSE (Log Space) : {rmse:.6f}\n"
        f"   MSE (Log Space)  : {mse:.6f}\n"
        f"   MAE (Log Space)  : {mae_log:.6f}\n"
        f"   MAE (Real Cycles): {mae_real:.2f}\n"
        f"   MAPE (Mean %)    : {mape:.4f}%\n\n"
        f"   Max APE          : {ape_max:.2f}%\n\n"
        f"2. Peak Values:\n"
        f"   Max Life (Log)   : {max_gt:.6f}\n"
        f"   Max Pred (Log)   : {max_pred:.6f}\n"
        f"   Max Error (Log)  : {max_error_log:.6f}\n"
        f"   Max Error (Real) : {max_error_real:.2f} cycles\n"
        f"   Max Life (Real)  : {max_gt_real:.2f} cycles\n"
        f"   Max Pred (Real)  : {max_pred_real:.2f} cycles\n"
    )

    txt_filename = f"summary_idx{sample_index}_{safe_group_id}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, "w") as f:
        f.write(report)

    print(f"   Saved summary stats to: {txt_path}")

    return r2, mse, max_error_log


def visualize_rollout(rollout_data, sample_index, output_dir):

    if rollout_data is None or len(rollout_data) == 0:
        print("No data to visualize.")
        return

    if sample_index >= len(rollout_data):
        print(f"Error: Index {sample_index} out of range (Max: {len(rollout_data)-1})")
        return

    sample = rollout_data[sample_index]

    mesh_pos = sample['mesh_pos']
    if hasattr(mesh_pos, 'cpu'):
        mesh_pos = mesh_pos.cpu().numpy()

    cells = sample['cells']
    if hasattr(cells, 'cpu'):
        cells = cells.cpu().numpy()

    target = sample['target'].flatten()
    prediction = sample['prediction'].flatten()

    group_id = sample.get('group_id', 'Unknown')

    print(f"Processing Sample {sample_index} (Group: {group_id})...")

    r2, mse, max_err = calculate_and_save_metrics(
        mesh_pos, target, prediction, group_id, sample_index, output_dir
    )

    num_cells = cells.shape[0]
    nodes_per_cell = cells.shape[1]

    padding = np.full((num_cells, 1), nodes_per_cell, dtype=cells.dtype)
    cells_vtk = np.hstack((padding, cells)).ravel()

    if nodes_per_cell == 3:
        cell_types = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
    elif nodes_per_cell == 4:
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
    elif nodes_per_cell == 8:
        cell_types = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    else:
        cell_types = np.full(num_cells, pv.CellType.POLYGON, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells_vtk, cell_types, mesh_pos)

    grid.point_data["Target"] = target
    grid.point_data["Prediction"] = prediction
    grid.point_data["Error"] = np.abs(target - prediction)

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2000, 600])

    plotter.set_background('white')

    common_args = dict(
        show_edges=False,
        smooth_shading=True
    )

    plotter.subplot(0, 0)
    plotter.add_text(f'Ground Truth\nMax: {target.max():.2f}', font_size=10)
    plotter.add_mesh(grid, scalars="Target", cmap='coolwarm', **common_args)

    plotter.subplot(0, 1)
    plotter.add_text(f'Prediction\nR2: {r2:.4f}', font_size=10)
    plotter.add_mesh(grid, scalars="Prediction", cmap='coolwarm', **common_args)

    plotter.subplot(0, 2)
    plotter.add_text(f'Absolute Error\nMax: {max_err:.4f}', font_size=10)
    plotter.add_mesh(grid, scalars="Error", cmap='coolwarm', **common_args)

    plotter.link_views()
    plotter.camera_position = 'iso'

    safe_group_id = str(group_id).replace("/", "_").replace("\\", "_")

    filename = f"plot_idx{sample_index}_{safe_group_id}.png"
    save_path = os.path.join(output_dir, filename)

    plotter.screenshot(save_path)
    plotter.close()

    print(f"   Saved visualization plot to: {save_path}")

    vtu_path = os.path.join(output_dir, f"mesh_idx{sample_index}.vtu")
    grid.save(vtu_path)

    print("-" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str,
    default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/full_regressor/regDGCNN_seg/step_shaft_tensile_torsion_enriched9/EXP_full_relative_loss_onecycle_k50_step_tensile_torsion_enriched9/2026-03-16_13-48-45/rollout/rollout_epoch_2000.pkl")

    parser.add_argument('--index', type=int, default=5)

    parser.add_argument('--output_dir', type=str,
    default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/full_regressor/regDGCNN_seg/step_shaft_tensile_torsion_enriched9/EXP_full_relative_loss_onecycle_k50_step_tensile_torsion_enriched9/2026-03-16_13-48-45/rollout_csv_pyvista")

    args = parser.parse_args()

    epoch_str = extract_epoch_from_filename(args.file)

    final_output_dir = os.path.join(args.output_dir, f"rollout_epoch_{epoch_str}")

    os.makedirs(final_output_dir, exist_ok=True)

    print(f"Saving results to: {final_output_dir}")

    data = load_rollout(args.file)

    visualize_rollout(data, sample_index=args.index, output_dir=final_output_dir)
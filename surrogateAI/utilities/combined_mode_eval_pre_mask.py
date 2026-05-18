import torch
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add the parent directory to the Python path to find other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.fatigue_model import Model as FatigueModel
from utilities.dataset import TrajectoryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def squeeze_data_frame(data_frame): 
    """Removes the batch dimension from tensors in a data dictionary."""
    for k, v in data_frame.items(): 
        if isinstance(v, torch.Tensor):
            data_frame[k] = torch.squeeze(v, 0) 
    return data_frame 

def load_model(checkpoint_dir, params):
    """Loads a trained model from a checkpoint directory."""
    print(f"Loading model from: {checkpoint_dir}")
    checkpoint_path = Path(checkpoint_dir)
    
    model = FatigueModel(params, core_model_name="regDGCNN_seg")
    
    model_file_path_str = str(checkpoint_path / "best_model_checkpoint")
    if not Path(model_file_path_str + "_learned_model.pth").exists():
        print(f"-> 'best_model_checkpoint' not found, trying 'final_model_checkpoint'...")
        model_file_path_str = str(checkpoint_path / "final_model_checkpoint")

    model.load_model(model_file_path_str)
    model.to(device)
    model.eval()
    return model

def main():
    """
    Main function to evaluate the combined model by pre-masking the input data
    before sending it to the respective expert models.
    """
    # --- USER ACTION REQUIRED ---
    class Args:
        classifier_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/classifier/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_class_shaft_low_extra_suffle_k80_1000/2025-08-10_15-16-48/checkpoint"
        lcf_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/lcf_regressor/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_lcf_shaft_low_extra_suffle_k40_2000/2025-08-10_12-59-25/checkpoint"
        hcf_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/hcf_regressor/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_hcf_extra_suffle_k5_1000/2025-08-14_11-26-58/checkpoint"
        
        k_classifier = 80 
        k_lcf = 40
        k_hcf = 5

        test_data = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/datasets/extracted_data/shaft_low_extra_.h5"
        output_dir = "./combined_test_results_pre_masking/kc_80_1000_klcf_40_2000_khcf_5_1000"
    
    args = Args()
    # --- END OF USER ACTION SECTION ---
    
    base_params = { 'output_size': 1 }
    
    print("--- Loading All Three Models ---")
    classifier = load_model(args.classifier_dir, {**base_params, 'purpose': 'classifier', 'k': args.k_classifier})
    lcf_regressor = load_model(args.lcf_dir, {**base_params, 'purpose': 'lcf_regressor', 'k': args.k_lcf})
    hcf_regressor = load_model(args.hcf_dir, {**base_params, 'purpose': 'hcf_regressor', 'k': args.k_hcf})

    print("\n--- Loading Test Data (full_eval mode) ---")
    test_dataset = TrajectoryDataset(args.test_data, split='val', mode='full_eval')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\n--- Starting Combined Evaluation (Pre-Masking Inputs) ---")
    all_final_predictions, all_ground_truths = [], []
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(f"Evaluating sample {i+1}/{len(test_loader)}...")
            frame = squeeze_data_frame(data)
            
            #saving cells and mesh postion of each node for visualization
            mesh_pos = frame['mesh_pos'].cpu().numpy()
            cells = frame['cells'].cpu().numpy()
            
            # Save the geometry to a separate, efficient .npz file
            geometry_path = output_path / f"geometry_sample_{i}.npz"
            np.savez_compressed(geometry_path, mesh_pos=mesh_pos, cells=cells)
            print(f"  -> Saved geometry to {geometry_path}")
            
            
            ground_truth_life = frame['fatigue_life_raw'].to(device)
            
            # --- Step 1: Classify the full frame to get the mask ---
            classifier_logits = classifier(frame, is_training=False)
            is_hcf_mask_1d = (classifier_logits > 0.0).squeeze().cpu()
            is_lcf_mask_1d = ~is_hcf_mask_1d

            # --- Step 2: Split the input data based on the mask ---
            lcf_frame = {key: val[is_lcf_mask_1d] for key, val in frame.items() if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == len(is_lcf_mask_1d)}
            hcf_frame = {key: val[is_hcf_mask_1d] for key, val in frame.items() if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == len(is_hcf_mask_1d)}

            # --- Step 3: Get predictions for each subset ---
            lcf_log10_preds = torch.tensor([], device=device)
            hcf_log10_preds = torch.tensor([], device=device)

            if lcf_frame and 'mesh_pos' in lcf_frame and lcf_frame['mesh_pos'].shape[0] > 0:
                lcf_log10_preds = lcf_regressor(lcf_frame, is_training=False)

            if hcf_frame and 'mesh_pos' in hcf_frame and hcf_frame['mesh_pos'].shape[0] > 0:
                hcf_log10_preds = hcf_regressor(hcf_frame, is_training=False)
            
            # --- Step 4: Reassemble the final prediction tensors (both scales) ---
            final_prediction = torch.zeros_like(ground_truth_life)
            final_log10_prediction = torch.zeros(len(is_lcf_mask_1d), device=device)

            final_prediction[is_lcf_mask_1d.to(device)] = 10**lcf_log10_preds
            final_prediction[is_hcf_mask_1d.to(device)] = 10**hcf_log10_preds
            
            # --- THIS IS THE FIX ---
            # Squeeze the prediction tensors to make them 1D before assigning them.
            final_log10_prediction[is_lcf_mask_1d.to(device)] = lcf_log10_preds.squeeze()
            final_log10_prediction[is_hcf_mask_1d.to(device)] = hcf_log10_preds.squeeze()
            # --- END OF FIX ---
            
            all_final_predictions.append(final_prediction.cpu())
            all_ground_truths.append(ground_truth_life.cpu())

            # --- Save the final combined (masked) prediction ---
            gt_np = ground_truth_life.cpu().numpy().flatten()
            pred_np_final = final_prediction.cpu().numpy().flatten()
            
            absolute_error = np.abs(gt_np - pred_np_final)
            percentage_error = (absolute_error / (gt_np + 1e-9)) * 100
            
            # Convert log tensors to numpy for the CSV file
            gt_log10_np = np.log10(np.maximum(gt_np, 1e-9)) # Calculate log10 from raw ground truth
            pred_log10_np = final_log10_prediction.cpu().numpy().flatten()

            df_final = pd.DataFrame({
                'node_index': np.arange(len(gt_np)),
                'ground_truth_log10_life': gt_log10_np,
                'predicted_log10_life': pred_log10_np,
                'ground_truth_life': gt_np,
                'final_predicted_life': pred_np_final,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'predicted_class_by_ml': is_hcf_mask_1d.cpu().numpy().astype(int)
            })

            csv_path_final = output_path / f"prediction_sample_{i}.csv"
            df_final.to_csv(csv_path_final, index=False)
            print(f"  -> Saved final combined results to {csv_path_final}")

    print("\n--- Overall Final Metrics ---")
    all_predictions_tensor = torch.cat(all_final_predictions)
    all_ground_truths_tensor = torch.cat(all_ground_truths)

    overall_mae = torch.nn.functional.l1_loss(all_predictions_tensor, all_ground_truths_tensor).item()
    print(f"Overall Mean Absolute Error (MAE): {overall_mae:,.2f}")

if __name__ == "__main__":
    main()

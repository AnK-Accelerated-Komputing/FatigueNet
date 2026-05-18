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
    Main function to run the combined evaluation using the trained ML classifier
    and save all intermediate and final predictions.
    """
    # --- USER ACTION REQUIRED ---
    # Set all the parameters for your evaluation run here.
    class Args:
        # Paths to your three trained models
        classifier_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/classifier/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_class_shaft_low_extra_suffle_k80_1000/2025-08-10_15-16-48/checkpoint"
        lcf_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/lcf_regressor/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_lcf_shaft_low_extra_suffle_k40_2000/2025-08-10_12-26-23/checkpoint"
        hcf_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/hcf_regressor/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_hcf_shaft_low_extra_suffle_k20_2000/2025-08-10_14-22-39/checkpoint"
        
        # 'k' value used to train EACH model.
        k_classifier = 80
        k_lcf = 40
        k_hcf = 20
        
        # Path to the H5 file containing the validation data
        val_data = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/datasets/extracted_data/shaft_low_extra_.h5"
        
        # Directory where the output CSV files will be saved
        output_dir = "./combined_eval_results"
    
    args = Args()
    # --- END OF USER ACTION SECTION ---
    
    base_params = { 'output_size': 1 }
    
    print("--- Loading All Three Models ---")
    classifier = load_model(args.classifier_dir, {**base_params, 'purpose': 'classifier', 'k': args.k_classifier})
    lcf_regressor = load_model(args.lcf_dir, {**base_params, 'purpose': 'lcf_regressor', 'k': args.k_lcf})
    hcf_regressor = load_model(args.hcf_dir, {**base_params, 'purpose': 'hcf_regressor', 'k': args.k_hcf})

    print("\n--- Loading Validation Data (full_eval mode) ---")
    val_dataset = TrajectoryDataset(args.val_data, split='val', mode='full_eval')
    if len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Exiting.")
        return
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"DataLoader created successfully. It contains {len(val_loader)} validation samples.")


    print("\n--- Starting Combined Evaluation with ML Classifier ---")
    all_final_predictions, all_ground_truths = [], []
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            print(f"Evaluating sample {i+1}/{len(val_loader)}...")
            frame = squeeze_data_frame(data)
            
            ground_truth_life = frame['fatigue_life_raw'].to(device)
            
            # Get log10 predictions from both models
            lcf_log10_preds = lcf_regressor(frame, is_training=False)
            hcf_log10_preds = hcf_regressor(frame, is_training=False)
            
            # Convert BOTH predictions back to the original life cycle scale
            lcf_preds = 10**lcf_log10_preds
            hcf_preds = 10**hcf_log10_preds

            # --- SAVE UNMASKED PREDICTIONS ---
            gt_np = ground_truth_life.cpu().numpy().flatten()

            lcf_pred_np = lcf_preds.cpu().numpy().flatten()
            df_lcf = pd.DataFrame({'node_index': np.arange(len(gt_np)), 'ground_truth_life': gt_np, 'unmasked_lcf_prediction': lcf_pred_np})
            csv_path_lcf = output_path / f"unmasked_lcf_prediction_sample_{i}.csv"
            df_lcf.to_csv(csv_path_lcf, index=False)
            print(f"  -> Saved LCF unmasked results to {csv_path_lcf}")

            hcf_pred_np = hcf_preds.cpu().numpy().flatten()
            df_hcf = pd.DataFrame({'node_index': np.arange(len(gt_np)), 'ground_truth_life': gt_np, 'unmasked_hcf_prediction': hcf_pred_np})
            csv_path_hcf = output_path / f"unmasked_hcf_prediction_sample_{i}.csv"
            df_hcf.to_csv(csv_path_hcf, index=False)
            print(f"  -> Saved HCF unmasked results to {csv_path_hcf}")
            # --- END OF SAVING UNMASKED ---

            # --- ML Classifier and Final Combination ---
            classifier_logits = classifier(frame, is_training=False)
            
            # --- THIS IS THE FIX ---
            # The comparison (>) already produces a boolean tensor with the correct
            # shape of [N, 1]. No squeeze() or unsqueeze() is needed.
            is_hcf_mask = (classifier_logits > 0.0)
            # --- END OF FIX ---
            
            final_prediction = torch.where(is_hcf_mask, hcf_preds, lcf_preds)
            
            all_final_predictions.append(final_prediction.cpu())
            all_ground_truths.append(ground_truth_life.cpu())

            # Save the final combined (masked) prediction
            pred_np_final = final_prediction.cpu().numpy().flatten()
            absolute_error = np.abs(gt_np - pred_np_final)
            percentage_error = (absolute_error / (gt_np + 1e-9)) * 100

            df_final = pd.DataFrame({
                'node_index': np.arange(len(gt_np)),
                'ground_truth_life': gt_np,
                'final_predicted_life': pred_np_final,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'predicted_class_by_ml': is_hcf_mask.cpu().numpy().flatten().astype(int)
            })
            csv_path_final = output_path / f"final_combined_prediction_sample_{i}.csv"
            df_final.to_csv(csv_path_final, index=False)
            print(f"  -> Saved final combined results to {csv_path_final}")

    print("\n--- Overall Final Metrics ---")
    all_predictions_tensor = torch.cat(all_final_predictions)
    all_ground_truths_tensor = torch.cat(all_ground_truths)

    overall_mae = torch.nn.functional.l1_loss(all_predictions_tensor, all_ground_truths_tensor).item()
    print(f"Overall Mean Absolute Error (MAE): {overall_mae:,.2f}")

if __name__ == "__main__":
    main()

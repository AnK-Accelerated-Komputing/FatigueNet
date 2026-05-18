import torch
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add the parent directory to the Python path to find other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now these imports should work correctly
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
    """Loads a single trained model from a checkpoint directory."""
    print(f"Loading model from: {checkpoint_dir}")
    checkpoint_path = Path(checkpoint_dir)
    
    # The 'purpose' in params is crucial for correct model initialization
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
    Main function to evaluate a single trained model (regressor OR classifier)
    on its corresponding validation or test dataset.
    """
    # --- USER ACTION REQUIRED ---
    # Set all the parameters for your evaluation run here.
    class Args:
        # Example for a regressor:
        # model_dir = "/path/to/your/lcf_regressor/.../checkpoint"
        # model_type = "lcf_regressor"
        
        # Example for a classifier:
        model_dir = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/output/hcf_regressor/regDGCNN_seg/shaft_low_extra_/EXPERIMENT_hcf_extra_suffle_k5_2000/2025-08-14_13-26-28/checkpoint"
        model_type = "hcf_regressor" # Can be 'lcf_regressor', 'hcf_regressor', or 'classifier'

        data_path = "/home/gd_user1/AnK/project_PINN/Project_Fatigue/Fatigue_Life_Combined/datasets/extracted_data/shaft_low_extra_.h5"
        data_split = "val"  # Can be 'val' or 'test'
        output_dir = "./single_model_evaluation/hcf_k5_2000" # A unique folder for the output
        neighbor_k = 5

    args = Args()
    # --- END OF USER ACTION SECTION ---

    params = {
        'k': args.neighbor_k,
        'output_size': 1,
        'purpose': args.model_type # Pass the purpose to the model
    }
    
    print(f"--- Loading {args.model_type.upper()} Model ---")
    model = load_model(args.model_dir, params)

    print(f"\n--- Loading '{args.data_split}' data for mode '{args.model_type}' ---")
    dataset = TrajectoryDataset(args.data_path, split=args.data_split, mode=args.model_type)
    if len(dataset) == 0:
        print(f"Error: Dataset for split '{args.data_split}' and mode '{args.model_type}' is empty. Exiting.")
        return
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\n--- Starting Evaluation on {len(dataset)} samples ---")
    all_predictions = []
    all_ground_truths = []
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            frame = squeeze_data_frame(data)
            
            # --- THIS IS THE CHANGE: Conditional logic for evaluation ---
            if 'regressor' in args.model_type:
                ground_truth = frame['fatigue_life'].to(device)
                prediction = model(frame, is_training=False)
                
                all_predictions.append(prediction.cpu())
                all_ground_truths.append(ground_truth.cpu())

                gt_np = ground_truth.cpu().numpy().flatten()
                pred_np = prediction.cpu().numpy().flatten()
                
                ground_truth_life = 10**gt_np
                predicted_life = 10**pred_np
                
                absolute_error = np.abs(ground_truth_life - predicted_life)
                percentage_error = (absolute_error / (ground_truth_life + 1e-9)) * 100

                df = pd.DataFrame({
                    'node_index': np.arange(len(gt_np)),
                    'ground_truth_log10_life': gt_np,
                    'predicted_log10_life': pred_np,
                    'ground_truth_life': ground_truth_life,
                    'predicted_life': predicted_life,
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error
                })

            elif args.model_type == 'classifier':
                ground_truth = frame['fatigue_class'].to(device)
                prediction_logits = model(frame, is_training=False)
                
                # Convert logits to class predictions (0 or 1)
                predicted_class = (prediction_logits > 0).float()
                
                all_predictions.append(predicted_class.cpu())
                all_ground_truths.append(ground_truth.cpu())

                gt_np = ground_truth.cpu().numpy().flatten()
                pred_np = predicted_class.cpu().numpy().flatten()

                df = pd.DataFrame({
                    'node_index': np.arange(len(gt_np)),
                    'ground_truth_class': gt_np.astype(int),
                    'predicted_class': pred_np.astype(int),
                    'is_correct': (gt_np == pred_np).astype(int)
                })
            # --- END OF CHANGE ---
            
            csv_path = output_path / f"single_model_eval_sample_{i}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> Evaluated sample {i+1}, saved detailed results to {csv_path}")

    # --- Calculate and Report Final Metrics ---
    all_predictions_tensor = torch.cat(all_predictions)
    all_ground_truths_tensor = torch.cat(all_ground_truths)

    if 'regressor' in args.model_type:
        print("\n--- Overall Evaluation Metrics (on log10 values) ---")
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        overall_mse = mse_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
        overall_mae = l1_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
        print(f"Overall MSE (on log10): {overall_mse:.6f}")
        print(f"Overall MAE (L1 Loss on log10): {overall_mae:.6f}")
        
    elif args.model_type == 'classifier':
        print("\n--- Overall Evaluation Metrics ---")
        accuracy = (all_predictions_tensor == all_ground_truths_tensor).float().mean().item()
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()

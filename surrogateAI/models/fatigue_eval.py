import torch
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Assuming these modules are in your Python path
from models.fatigue_model import Model as FatigueModel 
from utilities.dataset import TrajectoryDataset 
from train import squeeze_data # Ensure this matches your train script import path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, trajectory, num_steps=None):
    """Performs model rollouts and create stats."""
    prediction = model(trajectory, is_training=False)
    
    traj_ops = {
        'cells': trajectory['cells'],
        'mesh_pos': trajectory['mesh_pos'],
        'gt_fatigue_life': trajectory['fatigue_life'],
        'pred_fatigue_life': prediction,
    }
    return traj_ops

def load_model(checkpoint_dir, params):
    """Loads a trained model (classifier or regressor) from a checkpoint directory."""
    print(f"Loading model from: {checkpoint_dir}")
    checkpoint_path = Path(checkpoint_dir)
    
    # Re-create the model architecture.
    model = FatigueModel(params, core_model_name="regDGCNN_seg")
    
    # Find the best or final model file to load
    model_file_path_str = str(checkpoint_path / "best_model_checkpoint")
    if not Path(model_file_path_str + "_learned_model.pth").exists():
        print(f"'best_model_checkpoint' not found, trying 'best_model' or 'final_model_checkpoint'...")
        # Fallbacks depending on what train.py saved it as
        if Path(str(checkpoint_path / "best_model") + "_learned_model.pth").exists():
            model_file_path_str = str(checkpoint_path / "best_model")
        else:
            model_file_path_str = str(checkpoint_path / "final_model_checkpoint")

    # Load the state dict and the normalizer (if it's a regressor)
    model.load_model(model_file_path_str)
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

def main(args):
    """
    Main function to run the combined evaluation of the mixture-of-experts model.
    """
    # --- 1. Define Model Parameters ---
    # CRITICAL FIX: Input size is now 8!
    base_params = {
        'k': args.neighbor_k,
        'output_size': 1,
        'input_size': args.input_size, 
    }
    
    # --- 2. Load the Three Trained Models ---
    print("--- Loading Models ---")
    classifier_params = {**base_params, 'purpose': 'classifier'}
    classifier = load_model(args.classifier_dir, classifier_params)

    lcf_params = {**base_params, 'purpose': 'regressor'}
    lcf_regressor = load_model(args.lcf_dir, lcf_params)
    
    hcf_params = {**base_params, 'purpose': 'regressor'}
    hcf_regressor = load_model(args.hcf_dir, hcf_params)

    # --- 3. Load Test Data ---
    # Uses the 'full_eval' mode which now includes normals, curvature, and is_surface
    print("\n--- Loading Test Data ---")
    test_dataset = TrajectoryDataset(args.test_data, split='val', mode='full_eval')
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Exiting.")
        return
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 4. Run Evaluation Loop ---
    print("\n--- Starting Combined Evaluation ---")
    all_predictions = []
    all_ground_truths = []
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(f"Evaluating sample {i+1}/{len(test_loader)}...")
            frame = squeeze_data(data)
            
            ground_truth_life = frame['fatigue_life_raw'].to(device)

            # --- Step A: Classify every node ---
            classifier_logits = classifier(frame, is_training=False)
            
            # FIXED: Removed .squeeze() to prevent torch.where broadcasting crash
            is_hcf_mask = (classifier_logits > 0.0)
            
            # --- Step B: Get log10 predictions from both regressors ---
            # Output here is already denormalized to pure Log10 space by fatigue_model.py
            lcf_log10_preds = lcf_regressor(frame, is_training=False)
            hcf_log10_preds = hcf_regressor(frame, is_training=False)

            # --- Step C: Convert log10 predictions back to original scale (Real Cycles) ---
            lcf_preds = 10**lcf_log10_preds
            hcf_preds = 10**hcf_log10_preds

            # --- Step D: Combine predictions using the classifier's decision ---
            final_prediction = torch.where(is_hcf_mask, hcf_preds, lcf_preds)
            
            all_predictions.append(final_prediction.cpu())
            all_ground_truths.append(ground_truth_life.cpu())

            # --- Save detailed results for this sample to a CSV file ---
            gt_np = ground_truth_life.cpu().numpy().flatten()
            pred_np = final_prediction.cpu().numpy().flatten()
            abs_error = np.abs(gt_np - pred_np)
            
            df = pd.DataFrame({
                'node_index': np.arange(len(gt_np)),
                'ground_truth_life': gt_np,
                'predicted_life': pred_np,
                'absolute_error': abs_error,
                'predicted_class': is_hcf_mask.cpu().numpy().flatten().astype(int) # 0=LCF, 1=HCF
            })
            
            csv_path = output_path / f"evaluation_sample_{i}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> Saved detailed evaluation to {csv_path}")

    # --- 5. Calculate and Report Overall Metrics ---
    print("\n--- Overall Evaluation Metrics ---")
    all_predictions_tensor = torch.cat(all_predictions)
    all_ground_truths_tensor = torch.cat(all_ground_truths)

    mse_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()
    
    overall_mse = mse_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
    overall_mae = l1_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
    
    print(f"Overall MSE: {overall_mse:,.2f}")
    print(f"Overall MAE (L1 Loss): {overall_mae:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the combined fatigue life prediction model and export results to CSV.")
    
    parser.add_argument('--classifier_dir', type=str, required=True, help="Path to the trained classifier's checkpoint directory.")
    parser.add_argument('--lcf_dir', type=str, required=True, help="Path to the trained LCF regressor's checkpoint directory.")
    parser.add_argument('--hcf_dir', type=str, required=True, help="Path to the trained HCF regressor's checkpoint directory.")
    
    parser.add_argument('--test_data', type=str, required=True, help="Path to the H5 file containing test data.")
    parser.add_argument('--output_dir', type=str, default="./evaluation_results", help="Directory to save evaluation CSV files.")
    
    # FIXED: Default k changed to 20 to match your current training parameters
    parser.add_argument('--neighbor_k', type=int, default=20, help="Number of neighbors (k) used during training.")
    # FIXED: Added input_size argument defaulting to 8
    parser.add_argument('--input_size', type=int, default=8, help="Number of input features per node.")
    
    args = parser.parse_args()
    main(args)






















































"""

import torch
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Assuming these modules are in your Python path
from models.fatigue_model import Model as FatigueModel # Rename to avoid confusion
from utilities.dataset import TrajectoryDataset # This needs the 'full_eval' mode
from project_fatigue._fatigue_single.surrogateAI.train import squeeze_data # Re-using helper from train script

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, trajectory, num_steps=None):
    #Performs model rollouts and create stats.
    
    prediction = model(trajectory, is_training=False)
    
    traj_ops = {
        'cells': trajectory['cells'],
        'mesh_pos': trajectory['mesh_pos'],
        'gt_fatigue_life': trajectory['fatigue_life'],
        'pred_fatigue_life': prediction,
        #'region_node_type': trajectory['region_node_type']
    }
    
    return traj_ops

def load_model(checkpoint_dir, params):
    #Loads a trained model (classifier or regressor) from a checkpoint directory.
    print(f"Loading model from: {checkpoint_dir}")
    checkpoint_path = Path(checkpoint_dir)
    
    # Re-create the model architecture. The 'purpose' in params is crucial.
    model = FatigueModel(params, core_model_name="regDGCNN_seg")
    
    # Find the best or final model file to load
    model_file_path_str = str(checkpoint_path / "best_model_checkpoint")
    if not Path(model_file_path_str + "_learned_model.pth").exists():
        print(f"'best_model_checkpoint' not found, trying 'final_model_checkpoint'...")
        model_file_path_str = str(checkpoint_path / "final_model_checkpoint")

    # Load the state dict and the normalizer (if it's a regressor)
    model.load_model(model_file_path_str)
    model.to(device)
    model.eval() # Set to evaluation mode
    return model

def main(args):
    
    #Main function to run the combined evaluation of the mixture-of-experts model.
    
    # --- 1. Define Model Parameters ---
    # These params must match what was used during training.
    base_params = {
        'k': args.neighbor_k,
        'output_size': 1,
        'input_size': 3, # Using only mesh_pos
    }
    
    # --- 2. Load the Three Trained Models ---
    print("--- Loading Models ---")
    classifier_params = {**base_params, 'purpose': 'classifier'}
    classifier = load_model(args.classifier_dir, classifier_params)

    lcf_params = {**base_params, 'purpose': 'regressor'}
    lcf_regressor = load_model(args.lcf_dir, lcf_params)
    
    hcf_params = {**base_params, 'purpose': 'regressor'}
    hcf_regressor = load_model(args.hcf_dir, hcf_params)

    # --- 3. Load Test Data ---
    # We use a special 'full_eval' mode to get the complete, unfiltered data.
    print("\n--- Loading Test Data ---")
    test_dataset = TrajectoryDataset(args.test_data, split='val', mode='full_eval')
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Exiting.")
        return
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 4. Run Evaluation Loop ---
    print("\n--- Starting Combined Evaluation ---")
    all_predictions = []
    all_ground_truths = []
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(f"Evaluating sample {i+1}/{len(test_loader)}...")
            frame = squeeze_data(data)
            
            ground_truth_life = frame['fatigue_life_raw'].to(device)

            # --- Step A: Classify every node ---
            classifier_logits = classifier(frame, is_training=False)
            is_hcf_mask = (classifier_logits > 0.0).squeeze()
            
            # --- Step B: Get log10 predictions from both regressors ---
            lcf_log10_preds = lcf_regressor(frame, is_training=False)
            hcf_log10_preds = hcf_regressor(frame, is_training=False)

            # --- Step C: Convert log10 predictions back to original scale ---
            lcf_preds = 10**lcf_log10_preds
            hcf_preds = 10**hcf_log10_preds

            # --- Step D: Combine predictions using the classifier's decision ---
            final_prediction = torch.where(is_hcf_mask, hcf_preds, lcf_preds)
            
            all_predictions.append(final_prediction.cpu())
            all_ground_truths.append(ground_truth_life.cpu())

            # --- NEW: Save detailed results for this sample to a CSV file ---
            gt_np = ground_truth_life.cpu().numpy().flatten()
            pred_np = final_prediction.cpu().numpy().flatten()
            abs_error = np.abs(gt_np - pred_np)
            
            df = pd.DataFrame({
                'node_index': np.arange(len(gt_np)),
                'ground_truth_life': gt_np,
                'predicted_life': pred_np,
                'absolute_error': abs_error,
                'predicted_class': is_hcf_mask.cpu().numpy().astype(int) # 0=LCF, 1=HCF
            })
            
            csv_path = output_path / f"evaluation_sample_{i}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  -> Saved detailed evaluation to {csv_path}")

    # --- 5. Calculate and Report Overall Metrics ---
    print("\n--- Overall Evaluation Metrics ---")
    all_predictions_tensor = torch.cat(all_predictions)
    all_ground_truths_tensor = torch.cat(all_ground_truths)

    mse_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()
    
    overall_mse = mse_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
    overall_mae = l1_loss_fn(all_predictions_tensor, all_ground_truths_tensor).item()
    
    print(f"Overall MSE: {overall_mse:,.2f}")
    print(f"Overall MAE (L1 Loss): {overall_mae:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the combined fatigue life prediction model and export results to CSV.")
    
    parser.add_argument('--classifier_dir', type=str, required=True, help="Path to the trained classifier's checkpoint directory.")
    parser.add_argument('--lcf_dir', type=str, required=True, help="Path to the trained LCF regressor's checkpoint directory.")
    parser.add_argument('--hcf_dir', type=str, required=True, help="Path to the trained HCF regressor's checkpoint directory.")
    
    parser.add_argument('--test_data', type=str, required=True, help="Path to the H5 file containing test data.")
    parser.add_argument('--output_dir', type=str, default="./evaluation_results", help="Directory to save evaluation CSV files.")
    
    parser.add_argument('--neighbor_k', type=int, default=150, help="Number of neighbors (k) used during training.")
    
    args = parser.parse_args()
    main(args)

    """
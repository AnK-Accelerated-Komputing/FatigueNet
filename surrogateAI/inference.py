import os
from pathlib import Path
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np

# Ensure these imports work with your folder structure
from utilities.dataset import TrajectoryDataset
from models import fatigue_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def squeeze_data(data):
    """Removes batch dimension (1, N, ...) -> (N, ...) for models expecting single samples."""
    return {key: value.squeeze(0) for key, value in data.items() if isinstance(value, torch.Tensor)}

def pickle_save(path, data):
    """Saves data to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def main(args):
    print("--- Starting Inference ---")
    
    # 1. SETUP MODE
    if args.model_type == 'classifier':
        dataset_mode = 'classifier'
    else:
        dataset_mode = args.model_type

    # 2. LOAD DATASET
    print(f"Loading test dataset from: {args.test_data}")
    # Using 'val' or 'test' split depending on your dataset implementation
    test_dataset = TrajectoryDataset(args.test_data, split='val', mode=dataset_mode)
    
    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty.")
        return
        
    # Batch size must be 1 for point cloud models of varying sizes
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Total samples to process: {len(test_dataset)}")

    # 3. INITIALIZE MODEL
    print(f"Initializing {args.model_name}...")
    params = dict(
        purpose=args.model_type,
        output_size=args.output_size,
        k=args.neighbor_k,
        input_size=args.input_size
    )
    model = fatigue_model.Model(params, core_model_name=args.model_name).to(device)

    # 4. LOAD SAVED WEIGHTS & NORMALIZERS
    print(f"Loading checkpoint from: {args.model_path}")
    # The training script saves models as "model_epoch_X_learned_model.pth"
    # model.load_model() automatically appends the suffix, so we clean the string first
    model_path_clean = str(args.model_path).replace("_learned_model.pth", "")
    
    try:
        model.load_model(model_path_clean)
        print("Model weights and normalizers loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. INFERENCE LOOP
    model.eval()
    test_rollouts = []
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            frame = squeeze_data(data)
            
            # Forward pass (is_training=False automatically denormalizes output)
            output = model(frame, is_training=False)
            
            # Calculate loss for logging
            if 'regressor' in args.model_type:
                target = frame['fatigue_life'].to(device)
                loss_val = nn.functional.mse_loss(output[:, :1], target).item()
            else:
                target = frame['fatigue_class'].to(device)
                preds = (output > 0).float()
                loss_val = (preds == target).float().mean().item()

            # Safely extract group_id (the sample name)
            group_id = data.get('group_id', [f'sample_{idx}'])
            if isinstance(group_id, list):
                group_id = group_id[0]

            # Construct rollout dictionary (EXACT match to training format)
            sample_rollout = {
                'group_id': group_id,
                'mesh_pos': frame['mesh_pos'].cpu().numpy(),
                'cells': frame['cells'].cpu().numpy(),
                'prediction': output.cpu().numpy(),
                'target': target.cpu().numpy(),
                'val_loss': loss_val
            }
            test_rollouts.append(sample_rollout)
            
            print(f"[{idx+1}/{len(test_dataset)}] Processed {group_id} | Loss: {loss_val:.6f}")

    # 6. SAVE RESULTS
    output_filename = "inference_rollout.pkl"
    output_path = os.path.join(args.output_dir, output_filename)
    
    pickle_save(output_path, test_rollouts)
    
    print("\n" + "="*50)
    print(f"Inference Complete!")
    print(f"Successfully saved {len(test_rollouts)} samples to:")
    print(f"{output_path}")
    print("="*50)
    print("You can now run 'rollout_csv.py' on the folder containing this file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and save rollout.pkl")
    
    # --- CRITICAL PATHS ---
    # Path to the specific epoch checkpoint (e.g., .../checkpoint/best_model or .../checkpoint/model_epoch_2000)
    parser.add_argument('--model_path', type=str, default="/home/coldforging2/project_fatigue/OUTPUT/full_regressor/regDGCNN_seg/dogbone_dataset_enriched8/EXP_full_relative_loss_model_onecycle_K50/2026-03-12_08-13-01/checkpoint/model_epoch_660", help="Path to the model checkpoint (without _learned_model.pth)")
    
    # Path to the test/val .h5 file
    parser.add_argument('--test_data', type=str, default="/home/coldforging2/project_fatigue/_DATA/dogbone_dataset_enriched8.h5", help="Path to the .h5 test dataset")
    
    # Directory to save the inference_rollout.pkl
    parser.add_argument('--output_dir', type=str, default="/home/coldforging2/project_fatigue/OUTPUT/full_regressor/regDGCNN_seg/dogbone_dataset_enriched8/EXP_full_relative_loss_model_onecycle_K50/2026-03-12_08-13-01/inference_output", help="Directory to save the pkl file")
    
    # --- MODEL ARCHITECTURE PARAMS (Must match training!) ---
    parser.add_argument('--model_type', type=str, default="full_regressor", choices=['classifier', 'lcf_regressor', 'hcf_regressor', 'full_regressor'])
    parser.add_argument('--model_name', type=str, default="regDGCNN_seg")
    parser.add_argument('--input_size', type=int, default=8)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--neighbor_k', type=int, default=10)

    args = parser.parse_args()
    main(args)
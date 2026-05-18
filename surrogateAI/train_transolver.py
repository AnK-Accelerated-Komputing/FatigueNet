import os 
from pathlib import Path 
import pickle 
import time 
import datetime 
import argparse 
import json
import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np

from utilities.dataset import TrajectoryDataset 
from models import fatigue_model
#from models import transolver_adapter as fatigue_model 
#from models import mlp_adapter as fatigue_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# --- NEW JSON HELPER FUNCTIONS ---

def save_hyperparameters_json(log_dir, args, run_start_time):
    """
    Saves hyperparameters and training configuration to a JSON file at the start of training.
    """
    hyperparams = {
        'start_time': datetime.datetime.fromtimestamp(run_start_time).isoformat(),
        'start_time_epoch': run_start_time,
        'model_type': args.model_type,
        'model_name': args.model_name,
        'train_data': args.train_data,
        'val_data': args.val_data,
        'output_dir': args.output_dir,
        'experiment_id': args.experiment_id,
        'epochs': args.epochs,
        'base_lr': args.base_lr,
        'pct_start': args.pct_start,
        'batch_size': args.batch_size,
        #'neighbor_k': args.neighbor_k,

        # Transolver settings
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'n_head': args.n_head,
        'slice_num': args.slice_num,
        
        'optimizer': 'AdamW',
        'weight_decay': 1e-3,
        'input_size': args.input_size,
        'output_size': args.output_size,
        'scheduler': args.schedular,
        'device': str(device),
        'training_status': 'in_progress'
    }
    
    # Save as 'training_config.json' inside the log directory
    json_path = Path(log_dir) / 'training_config.json'
    with open(json_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Hyperparameters saved to {json_path}")
    return json_path

def update_training_results_json(json_path, train_loss_history, val_loss_history, 
                                  time_history, run_start_time, args):
    """
    Updates the JSON file with training results.
    """
    # Load existing config
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Calculate timing stats
    end_time = time.time()
    total_time = end_time - run_start_time
    
    # --- Process Training Losses (List of floats) ---
    train_losses = [float(loss) for loss in train_loss_history]
    
    if train_losses:
        min_train_loss = min(train_losses)
        min_train_loss_epoch = train_losses.index(min_train_loss) + 1
        max_train_loss = max(train_losses)
        mean_train_loss = sum(train_losses) / len(train_losses)
    else:
        min_train_loss = min_train_loss_epoch = max_train_loss = mean_train_loss = None
    
    # --- Process Validation Losses (List of tuples: (epoch, loss)) ---
    # We need to unpack them to calculate stats
    if val_loss_history:
        val_epochs = [x[0] for x in val_loss_history]
        val_values = [float(x[1]) for x in val_loss_history]
        
        min_eval_loss = min(val_values)
        # Find index of min loss, then get corresponding epoch from val_epochs
        min_idx = val_values.index(min_eval_loss)
        min_eval_loss_epoch = val_epochs[min_idx]
        
        max_eval_loss = max(val_values)
        max_idx = val_values.index(max_eval_loss)
        max_eval_loss_epoch = val_epochs[max_idx]
        
        mean_eval_loss = sum(val_values) / len(val_values)
    else:
        min_eval_loss = min_eval_loss_epoch = max_eval_loss = max_eval_loss_epoch = mean_eval_loss = None
    
    # Calculate average epoch time
    avg_epoch_time = sum(time_history) / len(time_history) if time_history else 0
    
    # Update config with results
    config.update({
        'last_update': datetime.datetime.fromtimestamp(end_time).isoformat(),
        'total_training_time_seconds': total_time,
        'total_training_time_formatted': str(datetime.timedelta(seconds=int(total_time))),
        
        # Training Stats
        'current_epoch': len(train_losses),
        'min_training_loss': min_train_loss,
        'min_training_loss_epoch': min_train_loss_epoch,
        'mean_training_loss': mean_train_loss,
        
        # Validation Stats
        'number_of_evaluations': len(val_loss_history),
        'min_evaluation_loss': min_eval_loss,
        'min_evaluation_loss_epoch': min_eval_loss_epoch,
        'max_evaluation_loss': max_eval_loss,
        'max_evaluation_loss_epoch': max_eval_loss_epoch,
        'mean_evaluation_loss': mean_eval_loss,
        
        # Timing Stats
        'avg_epoch_time_seconds': avg_epoch_time,
    })
    
    # Save updated config
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Optional print removed here to avoid spamming the console 
    # print(f"Training config updated. Min Val Loss: {min_eval_loss} (Epoch {min_eval_loss_epoch})")

# --- END NEW FUNCTIONS ---

def squeeze_data_frame(data_frame): 
    """Removes batch dimension (1, N, ...) -> (N, ...) for models expecting single samples."""
    for k, v in data_frame.items(): 
        if isinstance(v, torch.Tensor):
            data_frame[k] = torch.squeeze(v, 0) 
    return data_frame 

def squeeze_data(data):
    return {key: value.squeeze(0) for key, value in data.items() if isinstance(value, torch.Tensor)}

def pickle_save(path, data): 
    with open(path, 'wb') as f: 
        pickle.dump(data, f) 

'''
def loss_fn_regressor(inputs, network_output, model, mape_weight=0.1, safety_penalty=5.0, critical_focus=10.0): 
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()
    
    # 1. Base Values
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    prediction_normalized = network_output[:, :1] 
    pred_log_life = target_normalizer.inverse(prediction_normalized)
    
    # ---------------------------------------------------------
    # CONSTRAINT A: The Asymmetric Safety Mask
    # ---------------------------------------------------------
    # error > 0 means pred > target (Over-prediction = DANGEROUS)
    # error < 0 means pred < target (Under-prediction = SAFE / CONSERVATIVE)
    error = pred_log_life - target_log_life
    
    # Create a multiplier mask: 1.0 for safe predictions, 'safety_penalty' for dangerous ones
    # Using torch.where is numerically stable and keeps the computational graph intact
    asymmetric_weight = torch.where(error > 0, safety_penalty, 1.0)
    
    # ---------------------------------------------------------
    # CONSTRAINT B: Critical Node Focus (Spatial Weighting)
    # ---------------------------------------------------------
    # Fatigue failure is defined by the weakest link. We must force the network 
    # to prioritize nodes with LOW fatigue life (the notches/fillets).
    # We invert the target life so lower life = exponentially higher loss weight.
    # We subtract from max life (assume max log life is ~7.0 for scaling)
    max_log_life = 7.0 
    node_importance = 1.0 + critical_focus * torch.exp(-(target_log_life) / 2.0)
    
    # Combine the weights for each individual node
    combined_node_weights = asymmetric_weight * node_importance
    
    # ---------------------------------------------------------
    # LOSS CALCULATIONS (Applying the weights)
    # ---------------------------------------------------------
    # TERM 1: Weighted Log-Space MSE
    node_mse = (target_log_norm - prediction_normalized) ** 2
    loss_log_mse = torch.mean(combined_node_weights * node_mse)
    
    # TERM 2: Weighted Real-Space MAPE (Fixing the previous bug)
    target_real = torch.pow(10, target_log_life)
    pred_real = torch.pow(10, pred_log_life)
    
    node_mape = torch.abs((target_real - pred_real) / (target_real + 1e-6))
    loss_real_mape = torch.mean(combined_node_weights * node_mape)
    
    # COMBINE
    total_loss = loss_log_mse + (mape_weight * loss_real_mape)
    
    return total_loss
'''

'''
def loss_fn_regressor(inputs, network_output, model): 
    
    #1. Pure Log-Space MSE Loss.
    #Highly stable, guaranteed smooth gradients. No safety bias.
    
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()
    
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    prediction_normalized = network_output[:,:1] 
    
    loss = torch.mean((target_log_norm - prediction_normalized) ** 2) 
    return loss
'''
'''
def loss_fn_regressor(inputs, network_output, model): 
    
    #2. Asymmetric Log-Space MSE.
    #Stable math + Engineering safety. Penalizes dangerous over-predictions (2x).
    
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()
    
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    prediction_normalized = network_output[:,:1] 
    
    error = prediction_normalized - target_log_norm
    
    # Positive Error = Dangerous over-prediction
    safety_mask = torch.where(error > 0, 2.0, 1.0)
    loss = torch.mean(safety_mask * (error ** 2))
    
    return loss
'''
def loss_fn_regressor(inputs, network_output, model, mape_weight=0.1): 
    
    #3. Relative Hybrid Loss: Log-Space MSE (stability) + Real-Space MAPE (precision).
    #Forces the model to fix the "One Digit Off" error in real cycles.
    
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()
    
    # TERM A: Log-Space Loss
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    prediction_normalized = network_output[:,:1] 
    loss_log_mse = torch.mean((target_log_norm - prediction_normalized) ** 2)
    
    # TERM B: Real-Space Loss
    pred_log_life = target_normalizer.inverse(prediction_normalized)
    target_real = torch.pow(10, target_log_life)
    pred_real = torch.pow(10, pred_log_life)
    loss_real_mape = torch.mean(torch.abs((target_real - pred_real) / (target_real + 1e-6)))
    loss_log_relative = torch.mean(torch.abs(pred_log_life - target_log_life))
    # COMBINE
    total_loss = loss_log_mse + (mape_weight * loss_log_relative)
    return total_loss

'''
def loss_fn_regressor(inputs, network_output, model, mape_weight=0.1): 
    
    #3. Hybrid Loss: Log-Space MSE (stability) + Real-Space MAPE (precision).
    #Forces the model to fix the "One Digit Off" error in real cycles.
    
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()
    
    # TERM A: Log-Space Loss
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    prediction_normalized = network_output[:,:1] 
    loss_log_mse = torch.mean((target_log_norm - prediction_normalized) ** 2)
    
    # TERM B: Real-Space Loss
    pred_log_life = target_normalizer.inverse(prediction_normalized)
    target_real = torch.pow(10, target_log_life)
    pred_real = torch.pow(10, pred_log_life)
    loss_real_mape = torch.mean(torch.abs((target_real - pred_real) / (target_real + 1e-6)))
    
    # COMBINE
    total_loss = loss_log_mse + (mape_weight * loss_real_mape)
    return total_loss
'''

'''
def loss_fn_regressor(inputs, network_output, model):
    
    #4. Combined Ultimate Loss: Log-MSE + Real-MAPE + Heavy Asymmetric Safety.
    #The most aggressive loss function. Heavily penalizes dangerous predictions (5x).
    
    target_log_life = inputs['fatigue_life'].to(device)
    target_normalizer = model.get_output_life_normalizer()

    # Get Normalized Predictions
    target_log_norm = target_normalizer(target_log_life, accumulate=False)
    pred_log_norm = network_output[:,:1]
    
    # Un-normalize for real space math
    pred_log_life = target_normalizer.inverse(pred_log_norm)
    
    # TERM A: Asymmetric Log Space Error
    log_error = pred_log_norm - target_log_norm
    safety_mask = torch.where(log_error > 0, 5.0, 1.0)
    loss_safety_weighted = torch.mean(safety_mask * (log_error ** 2))

    # TERM B: Real Space Error (Precise)
    target_real = torch.pow(10, target_log_life)
    pred_real = torch.pow(10, pred_log_life)
    loss_real_mape = torch.mean(torch.abs((target_real - pred_real) / (target_real + 1e-6)))

    # COMBINE
    total_loss = loss_safety_weighted + 0.1 * loss_real_mape
    return total_loss
'''



def loss_fn_classifier(inputs, network_output, model):
    target_class = inputs['fatigue_class'].to(device)
    prediction_logits = network_output[:,:1]
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(prediction_logits, target_class)

def prepare_files_and_directories(output_dir, model_type, core_model, train_data_path, experiment_id, resume=False): 
    train_data_name = Path(train_data_path).stem
    run_base_dir = Path(output_dir) / model_type / core_model / train_data_name / f"EXP_{experiment_id}"
    
    if resume:
        if not run_base_dir.exists():
             raise ValueError(f"Cannot resume: Experiment directory {run_base_dir} does not exist.")
        
        subdirs = [d for d in run_base_dir.iterdir() if d.is_dir()]
        subdirs.sort()
        if not subdirs:
             raise ValueError(f"Cannot resume: No run directories found in {run_base_dir}")
        
        run_dir = subdirs[-1] # Pick the latest one
        print(f"Resuming from latest run directory: {run_dir}")
    else:
        # Create unique timestamped folder for this run
        run_create_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = run_base_dir / run_create_datetime
    
    checkpoint_dir = run_dir / 'checkpoint'
    log_dir = run_dir / 'log'
    rollout_dir = run_dir / 'rollout'
    
    for d in [checkpoint_dir, log_dir, rollout_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Directories set to: {run_dir}")
    return str(checkpoint_dir), str(log_dir), str(rollout_dir)

def main(args): 
    # Record Training Start Time
    run_start_time = time.time()

    # --- RESUME LOGIC SETUP ---
    start_epoch = 0 
    if args.resume:
        start_epoch = args.resume_epoch
        print(f"--- Resuming training from epoch {start_epoch} ---")
    
    end_epoch = args.epochs 

    # --- 1. SETUP MODE ---
    if args.model_type == 'classifier':
        print("--- Mode: Training CLASSIFIER ---")
        dataset_mode = 'classifier'
        loss_fn = loss_fn_classifier
        best_metric = -1.0 
    else: 
        print(f"--- Mode: Training {args.model_type.upper()} ---")
        dataset_mode = args.model_type
        loss_fn = loss_fn_regressor
        best_metric = float('inf') 

    # --- 2. LOAD DATASETS ---
    print(f"Loading datasets from: {args.train_data}")
    train_dataset = TrajectoryDataset(args.train_data, split='train', mode=dataset_mode) 
    val_dataset = TrajectoryDataset(args.val_data, split='val', mode=dataset_mode)
    
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty.")
        return
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False) if len(val_dataset) > 0 else None

    # --- 3. INITIALIZE MODEL ---
    core_model = args.model_name
    params = dict(
        purpose=args.model_type,
        output_size=args.output_size, 
        #k=args.neighbor_k,
        input_size=args.input_size,
        
        # New Transolver params
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_head=args.n_head,
        slice_num=args.slice_num
    ) 
    model =fatigue_model.Model(params, core_model_name=core_model).to(device)
    
    # Fit Normalizer (Crucial step)
    if 'regressor' in args.model_type:
        model.fit_normalizer(train_dataloader)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-3) 
    
    if args.schedular == "OneCycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=args.base_lr,
            epochs=end_epoch, steps_per_epoch=len(train_dataloader),
            pct_start=args.pct_start
        )
    else:  # cosine
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min
        )   

    # Prepare directories
    checkpoint_dir, log_dir, rollout_dir = prepare_files_and_directories(
        args.output_dir, args.model_type, core_model, args.train_data, args.experiment_id, args.resume
    ) 

    # --- SAVE INITIAL JSON CONFIG ---
    json_config_path = save_hyperparameters_json(log_dir, args, run_start_time)

    # --- 4. LOAD STATE IF RESUMING ---
    if args.resume:
        print(f"Loading checkpoint from: {args.resume_model_path}")
        resume_path_clean = str(args.resume_model_path).replace("_learned_model.pth", "")
        model.load_model(resume_path_clean)
        
        opt_path = Path(checkpoint_dir) / f"optimizer_epoch_{start_epoch}.pth"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path))
            print("Optimizer state loaded.")
        else:
            print(f"Warning: Optimizer checkpoint not found at {opt_path}. Starting fresh optimizer.")

        sch_path = Path(checkpoint_dir) / f"scheduler_epoch_{start_epoch}.pth"
        if sch_path.exists():
            scheduler.load_state_dict(torch.load(sch_path))
            print("Scheduler state loaded.")
        else:
            print(f"Warning: Scheduler checkpoint not found at {sch_path}. Starting fresh scheduler.")

    # --- Initialize Logging Lists ---
    train_loss_history = []
    val_loss_history = [] 
    lr_history = []
    time_history = []

    # --- 5. TRAINING LOOP ---
    print(f"Starting training from epoch {start_epoch} to {end_epoch}...")
    
    for epoch in range(start_epoch, end_epoch): 
        epoch_start_time = time.time()  
        model.train() 
        epoch_loss = 0.0
        
        for data in train_dataloader: 
            frame = squeeze_data_frame(data) 
            
            # NOTE: Normalization is now entirely handled inside fatigue_model.py's process_inputs!
            output = model(frame, is_training=True) 
            loss = loss_fn(frame, output, model)
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            if args.schedular == "OneCycle":
                scheduler.step()

            epoch_loss += loss.item()

        if args.schedular != "OneCycle":
            scheduler.step()    

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss_history.append(epoch_loss)
        lr_history.append(current_lr)
        time_history.append(epoch_duration)
        
        print(f"Epoch {epoch + 1}/{end_epoch} | Time: {epoch_duration:.2f}s | Train Loss: {epoch_loss:.6f}")

        # --- SAVE & VALIDATE EVERY 10 EPOCHS ---
        if (epoch + 1) % 10 == 0 or (epoch + 1) == end_epoch:
            print(f" >> Saving checkpoint, logs, and running validation...")
            
            # A. Save Regular Snapshot Checkpoint
            model_save_path = str(Path(checkpoint_dir) / f"model_epoch_{epoch+1}")
            model.save_model(model_save_path)
            
            # Save Optimizer & Scheduler
            torch.save(optimizer.state_dict(), Path(checkpoint_dir) / f"optimizer_epoch_{epoch+1}.pth")
            torch.save(scheduler.state_dict(), Path(checkpoint_dir) / f"scheduler_epoch_{epoch+1}.pth")
            
            # B. Run Validation & Collect Rollouts
            if val_loader:
                model.eval()
                val_losses = []
                epoch_rollouts = []  

                with torch.no_grad():
                    for data in val_loader:
                        frame = squeeze_data(data)
                        
                        # Forward pass: model(...) with is_training=False automatically denormalizes 
                        # the output so it returns the REAL Log10 Life directly.
                        output = model(frame, is_training=False) 
                        
                        if 'regressor' in args.model_type:
                            target = frame['fatigue_life'].to(device)
                            
                            # Because output is already real Log10 life, we compare it DIRECTLY
                            # to the target Log10 life. No normalizer scaling needed here.
                            loss_val = nn.functional.mse_loss(output[:,:1], target).item()
                        else:
                            target = frame['fatigue_class'].to(device)
                            preds = (output > 0).float()
                            loss_val = (preds == target).float().mean().item()
                        
                        val_losses.append(loss_val)

                        sample_rollout = {
                            'group_id': data.get('group_id', ['unknown'])[0] if isinstance(data.get('group_id'), list) else 'unknown',
                            'mesh_pos': frame['mesh_pos'].cpu().numpy(),
                            'cells': frame['cells'].cpu().numpy(),
                            'prediction': output.cpu().numpy(),
                            'target': target.cpu().numpy(),
                            'val_loss': loss_val
                        }
                        epoch_rollouts.append(sample_rollout)

                # C. Log Metrics and Update Best Model
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_loss_history.append((epoch + 1, avg_val_loss)) 
                
                is_best = False
                metric_name = ""
                
                if 'regressor' in args.model_type:
                    metric_name = "MSE (Log Life)"
                    if avg_val_loss < best_metric:
                        best_metric = avg_val_loss
                        is_best = True
                else:
                    metric_name = "Accuracy"
                    if avg_val_loss > best_metric:
                        best_metric = avg_val_loss
                        is_best = True
                
                print(f"    Validation {metric_name}: {avg_val_loss:.6f}")

                if is_best:
                    print(f"    *** New Best Model! ({metric_name}: {best_metric:.6f}) ***")
                    best_model_path = str(Path(checkpoint_dir) / "best_model")
                    model.save_model(best_model_path)

                rollout_file = Path(rollout_dir) / f"rollout_epoch_{epoch + 1}.pkl"
                pickle_save(str(rollout_file), epoch_rollouts)
                print(f"    Rollouts saved to: {rollout_file}")
                
                # --- UPDATE JSON CONFIG WITH LATEST STATS ---
                update_training_results_json(
                    json_config_path, 
                    train_loss_history, 
                    val_loss_history, 
                    time_history, 
                    run_start_time, 
                    args
                )

            # F. Save Logs
            log_data = {
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'lr': lr_history,
                'time': time_history
            }
            log_file = Path(log_dir) / f"log_epoch_{epoch + 1}.pkl"
            pickle_save(str(log_file), log_data)
            print(f"    Logs saved to: {log_file}")

    # --- FINAL UPDATES ---
    final_log_data = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'lr': lr_history,
        'time': time_history
    }
    final_log_file = Path(log_dir) / "final_training_log.pkl"
    pickle_save(str(final_log_file), final_log_data)
    
    # Update JSON one last time to mark completion
    update_training_results_json(
        json_config_path, 
        train_loss_history, 
        val_loss_history, 
        time_history, 
        run_start_time, 
        args
    )
    
    # Mark status as completed in JSON
    if os.path.exists(json_config_path):
        with open(json_config_path, 'r') as f:
            config = json.load(f)
        config['training_status'] = 'completed'
        with open(json_config_path, 'w') as f:
            json.dump(config, f, indent=4)

    print("Training Completed Successfully. Final log and JSON saved.")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train fatigue prediction model.") 
    
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['classifier', 'lcf_regressor', 'hcf_regressor', 'full_regressor'])
    
    parser.add_argument('--train_data', type=str, default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/step_shaft_torsion_enriched9.h5") 
    parser.add_argument('--val_data', type=str, default="/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/step_shaft_torsion_enriched9.h5") 
    parser.add_argument('--output_dir', type=str, default="/home/coldforging/cold_forging_2D_github/project_fatigue/surrogateAI/output") 
    
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--epochs', type=int, default=3500) 
    parser.add_argument('--base_lr', type=float, default=2e-4) 
    parser.add_argument('--pct_start', type=float, default=0.1)
    parser.add_argument('--experiment_id', type=str, default="full_relative_loss_step_torsion_enriched9") 
    
    #parser.add_argument('--neighbor_k', type=int, default=10) 
    parser.add_argument('--model_name', type=str, default="reg_DGCNN_seg", choices=["reg_DGCNN_seg","transolver", "mlp"]) 
    

    parser.add_argument('--input_size', type=int, default=9)
    parser.add_argument('--output_size', type=int, default=1) 
    parser.add_argument('--schedular', type=str, default="OneCycle", choices=["cosine", "OneCycle"])
        
    parser.add_argument('--T_0', type=int, default=50, help="Number of epochs for the first restart.")
    parser.add_argument('--T_mult', type=int, default=2, help="Multiplier for T_0 after every restart.")
    parser.add_argument('--eta_min', type=float, default=1e-6, help="Minimum learning rate.")

    # --- TRANSOLVER ARGUMENTS ---
    parser.add_argument('--n_hidden', type=int, default=128, help="Embedding dimension for Transolver")
    parser.add_argument('--n_layers', type=int, default=5, help="Number of Transformer blocks")
    parser.add_argument('--n_head', type=int, default=8, help="Number of Attention heads")
    parser.add_argument('--slice_num', type=int, default=64, help="Physics attention slices")

    # --- RESUME ARGUMENTS ---
    parser.add_argument('--resume', action='store_true', help="Set to resume training from a checkpoint.")
    parser.add_argument('--resume_epoch', type=int, default=1840, help="Epoch number to resume from.")
    parser.add_argument('--resume_model_path', type=str, default="/home/coldforging2/project_fatigue/OUTPUT/full_regressor/transolver/dogbone_torsion_tensile_enriched8/EXP_full_relative_loss_dogbone_torsion_tensile_enriched8/2026-03-16_00-40-52/checkpoint/model_epoch_1840")

    args = parser.parse_args() 
    main(args)
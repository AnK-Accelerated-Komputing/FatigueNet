import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def plot_training_loss(exp_dir, epoch, k):
    """
    Finds the training log from the NEW pipeline, then plots the training loss and learning rate.
    """
    # 1. Build Paths (Updated for New Pipeline)
    log_dir = os.path.join(exp_dir, 'log')
    
    # Try specific epoch file first: log_epoch_{epoch}.pkl
    # Note: In the new pipeline, files are named log_epoch_10.pkl, log_epoch_20.pkl, etc.
    train_pkl = os.path.join(log_dir, f'log_epoch_{epoch}.pkl')
    
    if not os.path.isfile(train_pkl):
        # Fallback: Try final log file
        train_pkl = os.path.join(log_dir, 'final_training_log.pkl')
        
        if not os.path.isfile(train_pkl):
            # Print helpful error listing what IS there
            found_files = os.listdir(log_dir) if os.path.exists(log_dir) else "Directory not found"
            raise FileNotFoundError(f"Training log not found.\nLooking for: {train_pkl}\nFound in dir: {found_files}")

    print(f"Loading log data from: {train_pkl}")

    # 2. Load Data
    with open(train_pkl, 'rb') as f:
        data = pickle.load(f)

    # 3. Extract Data (Updated Keys for New Pipeline)
    # New keys are 'train_loss' and 'lr'
    epoch_losses = [float(x) for x in data['train_loss']]
    lrs = data['lr']
    epochs = list(range(1, len(epoch_losses) + 1))

    # --- PLOTTING LOGIC (Same as Old) ---

    # Calculate min and max loss and find their epochs
    if epoch_losses:
        min_loss = min(epoch_losses)
        max_loss = max(epoch_losses)
        min_loss_epoch = epoch_losses.index(min_loss) + 1
        max_loss_epoch = epoch_losses.index(max_loss) + 1
        print(f"Min training loss: {min_loss:.6f} at epoch {min_loss_epoch}")
        print(f"Max training loss: {max_loss:.6f} at epoch {max_loss_epoch}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, epoch_losses, marker='o', linestyle='-', label='Total Train Loss', color='blue', markersize=4, zorder=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, linestyle='--', label='Learning Rate', color='red')
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add annotations to the plot
    if epoch_losses:
        ax1.annotate(f'Min Loss: {min_loss:.4f}', 
                     xy=(min_loss_epoch, min_loss), 
                     xytext=(min_loss_epoch, min_loss + (max_loss - min_loss) * 0.1),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     ha='center', color='green')
        ax1.annotate(f'Max Loss: {max_loss:.4f}', 
                     xy=(max_loss_epoch, max_loss), 
                     xytext=(max_loss_epoch, max_loss - (max_loss - min_loss) * 0.1),
                     arrowprops=dict(facecolor='purple', shrink=0.05),
                     ha='center', color='purple')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Title
    ax1.set_title(f'Training Loss & Learning Rate (up to epoch {len(epochs)}) and k={k}')
    ax1.grid(True)

    # Save
    save_dir = os.path.join(exp_dir, 'plots_lr')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'training_loss_epoch_{epoch}_k{k}.png')
    loss_per_epoch_path = os.path.join(save_dir, f'training_loss_per_epoch_{epoch}_k{k}.txt')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved training plot to {out_path}")

    # --- TEXT REPORT GENERATION (Restored Original Logic) ---
    print("\nTraining Loss per Epoch:")
    # Clear the file before starting to append
    if os.path.exists(loss_per_epoch_path):
        os.remove(loss_per_epoch_path)
        
    with open(loss_per_epoch_path, 'w') as f:
         pass # Create empty file

    for i, loss in enumerate(epoch_losses, start=1):
        if i % 10 == 0:  # Changed to 10 because new pipeline saves every 10 epochs
            print(f"Epoch [{i}/{len(epoch_losses)}], Loss: {loss:.6f}")
            with open(loss_per_epoch_path, 'a') as f:
                f.write(f"Epoch {i}: Loss {loss:.6f}\n")
    
    # Append the summary to the end of the text file
    if epoch_losses:
        with open(loss_per_epoch_path, 'a') as f:
            f.write("\n--- Summary ---\n")
            f.write(f"Minimum Training Loss: {min_loss} (at Epoch {min_loss_epoch})\n")
            f.write(f"Maximum Training Loss: {max_loss} (at Epoch {max_loss_epoch})\n")
            
    print(f"Full training loss history and summary saved to {loss_per_epoch_path}")

def main():
    # --- USER ACTION REQUIRED: Update these for your current run ---
    # Path to the specific experiment folder (containing 'log', 'checkpoint' folders)
    exp_dir = '/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/transolver/step_shaft_torsion_enriched9/EXP_full_hybrid_relative_logspace_loss_onecycle_transolver/2026-03-05_15-07-04'
    
    epoch = 3500 # Epoch number to visualize (e.g., 10, 20... or the final epoch)
    k = 0      # Neighbor k used in training (just for the plot title)
    
    plot_training_loss(exp_dir, epoch, k)
    
if __name__ == '__main__':
    main()
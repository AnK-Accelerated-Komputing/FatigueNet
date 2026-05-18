import os
import matplotlib.pyplot as plt
import pickle

def plot_validation_loss(exp_dir, epoch, k):
    """
    Plots the validation loss history from the new training pipeline logs.
    """
    # 1. Build Paths
    log_dir = os.path.join(exp_dir, 'log')
    
    # Try specific epoch file first, then fallback to final log
    log_file = os.path.join(log_dir, f'log_epoch_{epoch}.pkl')
    if not os.path.isfile(log_file):
        log_file = os.path.join(log_dir, 'final_training_log.pkl')
        if not os.path.isfile(log_file):
             raise FileNotFoundError(f"Validation log not found in: {log_dir}")

    print(f"Loading validation data from: {log_file}")

    # 2. Load Data
    with open(log_file, 'rb') as f:
        data = pickle.load(f)

    # 3. Extract Validation Data
    # Format in new pipeline is list of tuples: [(epoch, loss), (epoch, loss), ...]
    val_history = data['val_loss']
    
    if not val_history:
        print("No validation data found in this log.")
        return

    # Unpack into separate lists for plotting
    epochs = [x[0] for x in val_history]
    losses = [float(x[1]) for x in val_history]

    # 4. Calculate Statistics
    min_loss = min(losses)
    max_loss = max(losses)
    min_loss_epoch = epochs[losses.index(min_loss)]
    max_loss_epoch = epochs[losses.index(max_loss)]

    print(f"Min Val Loss: {min_loss:.6f} at Epoch {min_loss_epoch}")

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, losses, marker='o', linestyle='-', linewidth=2, markersize=6, color='orange', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE/Accuracy)')
    ax.set_title(f'Validation Loss Trend (k={k})')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Annotations for Min/Max
    ax.annotate(f'Min: {min_loss:.4f}', 
                xy=(min_loss_epoch, min_loss), 
                xytext=(min_loss_epoch, min_loss + (max_loss - min_loss)*0.1),
                arrowprops=dict(facecolor='green', shrink=0.05),
                ha='center', color='green', fontweight='bold')

    ax.legend()

    # 6. Save Plot
    save_dir = os.path.join(exp_dir, 'plots_val')
    os.makedirs(save_dir, exist_ok=True)
    
    out_img_path = os.path.join(save_dir, f'val_loss_epoch_{epoch}_k{k}.png')
    out_txt_path = os.path.join(save_dir, f'val_loss_epoch_{epoch}_k{k}.txt')
    
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_img_path}")

    # 7. Save Text Summary
    with open(out_txt_path, 'w') as f:
        f.write(f"--- Validation Loss Summary (k={k}) ---\n")
        f.write(f"Log File: {log_file}\n")
        f.write(f"Min Loss: {min_loss:.6f} (Epoch {min_loss_epoch})\n")
        f.write(f"Max Loss: {max_loss:.6f} (Epoch {max_loss_epoch})\n")
        f.write("\n--- History ---\n")
        for ep, loss in zip(epochs, losses):
            f.write(f"Epoch {ep}: {loss:.6f}\n")
            
    print(f"Saved summary text to: {out_txt_path}")

def main():
    # --- USER CONFIGURATION ---
    exp_dir = '/home/godelblock/PINN/project_fatigue/_fatigue_single/output/full_regressor/regDGCNN_seg/fatigue_dataset/EXP_full_run_k100/2026-02-10_05-36-20'
    epoch = 950
    k = 100
    
    plot_validation_loss(exp_dir, epoch, k)

if __name__ == '__main__':
    main()
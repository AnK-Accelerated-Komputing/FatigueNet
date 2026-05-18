import h5py
import numpy as np
import pyvista as pv

"""
# Update this path to your actual H5 file
h5_path = "/home/godelblock/PINN/project_fatigue/_fatigue_single/datasets/raw_data/fatigue_dataset.h5"
THRESHOLD=10**5  # 100,000 cycles
print(f"Checking data range in: {h5_path}")

try:
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        all_lives = []
        
        print(f"Found {len(keys)} groups.")
        
        for key in keys:
            life = f[key]['fatigue_life'][:]
            all_lives.append(life)
            
        all_lives = np.concatenate(all_lives)
        
        min_life = np.min(all_lives)
        max_life = np.max(all_lives)
        
        print(f"\n--- DATA STATISTICS ---")
        print(f"Minimum Life: {min_life:.2f}")
        print(f"Maximum Life: {max_life:.2f}")
        print(f"Threshold:    100000.00")
        
        count_lcf = np.sum(all_lives <= THRESHOLD)
        count_hcf = np.sum(all_lives > THRESHOLD)
        
        print(f"\nLCF Samples (<= 10^5): {count_lcf} ({(count_lcf/len(all_lives))*100:.2f}%)")
        print(f"HCF Samples (> 10^5):  {count_hcf} ({(count_hcf/len(all_lives))*100:.2f}%)")
        
        if count_lcf == 0:
            print("\nCONCLUSION: Your dataset contains NO Low Cycle Fatigue data.")
            print("Try running with --model_type hcf_regressor or full_regressor.")

except Exception as e:
    print(f"Error reading file: {e}")

"""


# Point this to your enriched H5 file
H5_PATH = r"/home/godelblock/PINN/project_fatigue/200 new/fatigue_dataset_enriched.h5"

with h5py.File(H5_PATH, 'r') as f:
    # Let's just look at the first CAD part
    group = f['group_0']
    
    mesh_pos = group['mesh_pos'][:]
    is_surface = group['is_surface'][:]
    
# Separate the coordinates based on the mask
surface_nodes = mesh_pos[is_surface.flatten() == 1.0]
interior_nodes = mesh_pos[is_surface.flatten() == 0.0]

print(f"Total Nodes: {len(mesh_pos)}")
print(f"Surface Nodes: {len(surface_nodes)}")
print(f"Interior Nodes: {len(interior_nodes)}")

# --- VISUALIZE ---
plotter = pv.Plotter()

# Plot interior nodes in RED
plotter.add_points(interior_nodes, color='red', point_size=4, label='Interior Nodes (0)')

# Plot surface nodes in BLUE (slightly transparent so we can see inside)
plotter.add_points(surface_nodes, color='blue', point_size=6, opacity=0.3, label='Surface Nodes (1)')

plotter.add_legend()
plotter.show()
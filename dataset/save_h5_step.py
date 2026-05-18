import os
import glob
import numpy as np
import h5py
import re

# --- HELPER FOR NATURAL SORTING ---
def natural_keys(text):
    """
    Splits a string into a list of integers and text.
    Example: 'file_10' -> ['file_', 10]
    This forces Python to treat numbers as numbers, not text.
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def parse_mesh_dat(dat_file_path):
    """
    Parses ANSYS mesh.dat file to extract node coordinates and element connectivity.
    """
    nodes = []
    cells = []
    
    in_nodes = False
    in_elements = False
    
    with open(dat_file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith("nblock"):
            in_nodes = True
            in_elements = False
            continue
        
        if line.lower().startswith("eblock"):
            in_elements = True
            in_nodes = False
            continue
            
        if line == "-1":
            in_nodes = False
            in_elements = False
            continue
            
        if in_nodes:
            if line.startswith('('): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    nodes.append((nid, x, y, z))
                except ValueError:
                    continue 

        if in_elements:
            if line.startswith('('): continue
            parts = line.split()
            if len(parts) >= 9:
                try:
                    # --- ONLY CHANGED THIS LINE ---
                    # dict.fromkeys() removes duplicates while keeping the exact original order
                    node_ids = list(dict.fromkeys([int(p) for p in parts[-8:]]))
                    # ------------------------------
                    cells.append(node_ids)
                except ValueError:
                    continue

    nodes.sort(key=lambda x: x[0]) 
    node_map = {node[0]: i for i, node in enumerate(nodes)}
    mesh_pos = np.array([n[1:] for n in nodes], dtype=np.float32)
    
    mapped_cells = []
    for cell in cells:
        try:
            mapped_cell = [node_map[nid] for nid in cell]
            mapped_cells.append(mapped_cell)
        except KeyError:
            continue
            
    cells_array = np.array(mapped_cells, dtype=np.int32)
    return mesh_pos, cells_array, node_map

def parse_result_file(filepath, node_map, num_nodes):
    data = np.zeros((num_nodes, 1), dtype=np.float32)
    if not filepath or not os.path.exists(filepath):
        return data

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2: continue
            try:
                nid = int(parts[0])
                val = float(parts[-1]) 
                if nid in node_map:
                    idx = node_map[nid]
                    data[idx] = val
            except ValueError:
                continue 
    return data

def find_file_by_pattern(folder, pattern):
    try:
        files = os.listdir(folder)
        for f in files:
            if re.search(pattern, f, re.IGNORECASE):
                return os.path.join(folder, f)
    except FileNotFoundError:
        return None
    return None

def process_dataset(raw_data_dir, output_h5_path):
    
    # 1. Get all folders
    subfolders = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]
    
    # 2. SORT THEM NATURALLY
    # This fixes the "1, 10, 2" ordering issue
    subfolders.sort(key=natural_keys)
    
    print(f"Found {len(subfolders)} simulation folders.")
    
    with h5py.File(output_h5_path, 'w') as h5f:
        
        group_id = 0 
        
        for folder in subfolders:
            folder_name = os.path.basename(folder)
            
            # Check for mesh.dat
            mesh_file = os.path.join(folder, "mesh.dat")
            if not os.path.exists(mesh_file):
                continue
                
            try:
                mesh_pos, cells, node_map = parse_mesh_dat(mesh_file)
            except Exception as e:
                print(f"  [Error] Failed to parse mesh in {folder_name}: {e}")
                continue
                
            num_nodes = len(mesh_pos)
            
            # Find result files
            path_life = find_file_by_pattern(folder, r"Life_.*\.txt")
            path_damage = find_file_by_pattern(folder, r"Damage_.*\.txt")
            path_stress = find_file_by_pattern(folder, r"Equivalent_Alternating_Stress_.*\.txt")
            path_safety = find_file_by_pattern(folder, r"Safety_Factor_.*\.txt")
            
            # Parse results
            life_data = parse_result_file(path_life, node_map, num_nodes)
            damage_data = parse_result_file(path_damage, node_map, num_nodes)
            stress_data = parse_result_file(path_stress, node_map, num_nodes)
            safety_data = parse_result_file(path_safety, node_map, num_nodes)
            
            # Save to H5
            group_name = f"group_{group_id}" 
            grp = h5f.create_group(group_name)
            
            grp.attrs['original_folder_name'] = folder_name
            
            # Saving RAW (No compression)
            grp.create_dataset("mesh_pos", data=mesh_pos)
            grp.create_dataset("cells", data=cells)
            grp.create_dataset("fatigue_life", data=life_data)
            grp.create_dataset("damage", data=damage_data)
            grp.create_dataset("eq_stress", data=stress_data)
            grp.create_dataset("safety_factor", data=safety_data)
            
            print(f"Processed {group_name}: {folder_name}")
            
            group_id += 1 
            
    print(f"\nSuccessfully processed {group_id} datasets.")
    print(f"Saved to: {output_h5_path}")

if __name__ == "__main__":
    RAW_DATA_DIR = r"/home/godelblock/PINN/project_fatigue/200 new/200 new"
    OUTPUT_FILE = r"/home/godelblock/PINN/project_fatigue/200 new/fatigue_dataset.h5"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_dataset(RAW_DATA_DIR, OUTPUT_FILE)
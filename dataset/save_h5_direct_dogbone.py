import os
import h5py
import numpy as np
import pandas as pd
import re

# --- HELPER FOR NATURAL SORTING ---
def natural_keys(text):
    """
    Splits a string into a list of integers and text.
    Example: 'Fatigue_Results_10' -> ['Fatigue_Results_', 10]
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def parse_nodes_and_base_values(filepath):
    """
    Reads a result file (fatiguelife.txt) to extract:
    1. Node Coordinates (X, Y, Z).
    2. Node Mapping (NodeID -> Index).
    3. The values in this file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    # Sort by NodeID
    df = df.sort_values(by='NodeID')
    
    # Create Mapping: NodeID -> Array Index
    node_ids = df['NodeID'].values
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    
    # Extract Mesh Position (N, 3)
    mesh_pos = df[['X', 'Y', 'Z']].values.astype(np.float32)
    
    # Extract Values (N, 1)
    values = df['Value'].values.astype(np.float32).reshape(-1, 1)
    
    return mesh_pos, values, node_map

def parse_connectivity(filepath, node_map):
    """
    Parses mesh.dat for element connectivity.
    Ignores the misleading header and reads all columns.
    Returns an (N_elements, 4) matrix.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # SKIP the first row (header "ElementID,NodeIDs")
        # HEADER=None tells pandas to assign index 0, 1, 2... to columns automatically
        df = pd.read_csv(filepath, skiprows=1, header=None)
    except Exception as e:
        print(f"Error reading mesh.dat: {e}")
        return np.array([])

    cells = []
    
    # Column 0 is ElementID (we ignore it).
    # Columns 1 onwards are NodeIDs.
    # We extract the values as a numpy array directly for speed.
    if df.shape[1] < 2:
        print("Error: mesh.dat has insufficient columns.")
        return np.array([])

    # Get only the NodeID columns (drop the first column)
    raw_node_ids = df.iloc[:, 1:].values
    
    # Iterate through rows to map NodeIDs to indices
    # We keep the row structure!
    for row in raw_node_ids:
        try:
            # Map every node ID in this element row
            mapped_row = [node_map[int(nid)] for nid in row if not pd.isna(nid)]
            cells.append(mapped_row)
        except KeyError:
            # If a node ID doesn't exist in our map (rare), we skip this element
            continue

    # Convert to numpy array (N_elements, Nodes_per_element)
    return np.array(cells, dtype=np.int32)

def parse_scalar_file(filepath, node_map, num_nodes):
    """
    Parses other result files using the existing node map.
    """
    data = np.zeros((num_nodes, 1), dtype=np.float32)
    
    if not os.path.exists(filepath):
        # Optional: Print warning only once or skip
        return data

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    for row in df.itertuples(index=False):
        nid = int(row.NodeID)
        val = float(row.Value)
        
        if nid in node_map:
            idx = node_map[nid]
            data[idx] = val
            
    return data

def process_dataset(raw_data_dir, output_h5_path):
    
    # 1. Get all subfolders
    subfolders = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]
    
    # 2. Sort naturally
    subfolders.sort(key=lambda x: natural_keys(os.path.basename(x)))
    
    print(f"Found {len(subfolders)} simulation folders.")
    
    with h5py.File(output_h5_path, 'w') as h5f:
        
        group_id = 0 
        
        for folder in subfolders:
            folder_name = os.path.basename(folder)
            
            # Define File Paths
            file_life = os.path.join(folder, "fatiguelife.txt")
            file_mesh = os.path.join(folder, "mesh.dat")
            file_damage = os.path.join(folder, "damage.txt")
            file_stress = os.path.join(folder, "equ_stress.txt")
            file_safety = os.path.join(folder, "factor_of_safety.txt")
            
            # Check essential files
            if not os.path.exists(file_life) or not os.path.exists(file_mesh):
                continue
                
            print(f"Processing Group {group_id}: {folder_name} ...", end=" ")

            try:
                # 1. Parse Nodes and Fatigue Life (Base)
                mesh_pos, life_data, node_map = parse_nodes_and_base_values(file_life)
                num_nodes = len(mesh_pos)
                
                # 2. Parse Connectivity (FIXED)
                cells = parse_connectivity(file_mesh, node_map)
                
                # 3. Parse Other Scalars
                damage_data = parse_scalar_file(file_damage, node_map, num_nodes)
                stress_data = parse_scalar_file(file_stress, node_map, num_nodes)
                safety_data = parse_scalar_file(file_safety, node_map, num_nodes)
                
                # 4. Save to H5
                group_name = f"group_{group_id}" 
                grp = h5f.create_group(group_name)
                
                grp.attrs['original_folder_name'] = folder_name
                
                grp.create_dataset("mesh_pos", data=mesh_pos)
                grp.create_dataset("cells", data=cells)
                grp.create_dataset("fatigue_life", data=life_data)
                grp.create_dataset("damage", data=damage_data)
                grp.create_dataset("eq_stress", data=stress_data)
                grp.create_dataset("safety_factor", data=safety_data)
                
                print(f"Done. (Cells shape: {cells.shape})")
                group_id += 1
                
            except Exception as e:
                print(f" [ERROR] {e}")
                continue

    print(f"\nSuccessfully processed {group_id} datasets.")
    print(f"Saved to: {output_h5_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    RAW_DATA_DIR = r"/home/godelblock/PINN/project_fatigue/_DATA/step_shaft_tensile_torsion"
    OUTPUT_FILE = r"/home/godelblock/PINN/project_fatigue/_DATA/step_shaft_tensile_torsion.h5"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_dataset(RAW_DATA_DIR, OUTPUT_FILE)
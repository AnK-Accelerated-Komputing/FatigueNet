import os
import h5py
import numpy as np
import pyvista as pv
import time

def compute_exact_geometry(mesh_pos, cells_array, step_z_location):
    """
    Computes exact normals, mean curvature, and a surface mask 
    using true mesh topology via PyVista (VTK).
    """
    num_nodes = mesh_pos.shape[0]
    num_cells = cells_array.shape[0]
    nodes_per_cell = cells_array.shape[1]
    
    # 1. Format the cells for VTK/PyVista
    # VTK requires a padding number indicating how many nodes are in each cell.
    # For a 4-node Tetrahedron, it expects: [4, n1, n2, n3, n4, 4, n1, n2, n3, n4...]
    padding = np.full((num_cells, 1), nodes_per_cell, dtype=np.int32)
    vtk_cells = np.hstack((padding, cells_array)).flatten()
    
    # 2. Determine VTK cell type
    # 10 corresponds to vtkTetra (4-node tetrahedron). 
    # (If you ever use Hexahedrons with 8 nodes, this changes to 12)
    if nodes_per_cell == 4:
        cell_type = np.full(num_cells, 10, dtype=np.uint8)
    else:
        raise ValueError(f"Expected 4 nodes for Tetrahedrons, but got {nodes_per_cell}")
    
    # 3. Build the 3D Unstructured Grid
    grid = pv.UnstructuredGrid(vtk_cells, cell_type, mesh_pos)
    
    # 4. Extract the exact outer surface (The "Skin")
    surface = grid.extract_surface()
    
    # 5. Compute flawless point normals on the surface
    surface = surface.compute_normals(point_normals=True, cell_normals=False, 
                                      auto_orient_normals=True)
    
    # 6. Compute mean curvature on the surface
    curvature_vals = surface.curvature(curv_type='mean')
    
    # 7. Map the surface features back to the original full node list
    # Initialize arrays with zeros (Interior nodes remain exactly zero!)
    final_normals = np.zeros((num_nodes, 3), dtype=np.float32)
    final_curvature = np.zeros((num_nodes, 1), dtype=np.float32)
    is_surface = np.zeros((num_nodes, 1), dtype=np.float32) # The Surface Mask
    
    # PyVista keeps track of the original node IDs when it extracts the surface
    original_node_ids = surface["vtkOriginalPointIds"]
    
    # Inject the computed values into the correct indices for the surface nodes
    final_normals[original_node_ids] = surface["Normals"]
    final_curvature[original_node_ids, 0] = curvature_vals
    is_surface[original_node_ids, 0] = 1.0 # 1 = Surface Node, 0 = Interior Node
    
    # --- NEW ADDITION START ---
    # 8. Compute Absolute Distance to Step along Z-Axis
    dist_to_step = np.abs(mesh_pos[:, 2] - step_z_location).astype(np.float32)
    dist_to_step = dist_to_step.reshape(-1, 1) # Reshape to (num_nodes, 1)
    # --- NEW ADDITION END ---
    
    return final_normals, final_curvature, is_surface, dist_to_step


def enrich_h5_dataset(base_h5_path, output_h5_path, step_z_location):
    print(f"Reading base dataset from: {base_h5_path}")
    print(f"Creating enriched dataset at: {output_h5_path}")
    print(f"Adding dist_to_step feature for Z = {step_z_location}")
    
    start_time = time.time()
    
    with h5py.File(base_h5_path, 'r') as base_h5, h5py.File(output_h5_path, 'w') as out_h5:
        
        group_names = list(base_h5.keys())
        total_groups = len(group_names)
        
        for idx, group_name in enumerate(group_names):
            base_grp = base_h5[group_name]
            out_grp = out_h5.create_group(group_name)
            
            # 1. Copy original attributes (like folder name)
            for attr_name, attr_val in base_grp.attrs.items():
                out_grp.attrs[attr_name] = attr_val
                
            # 2. Copy all original datasets (mesh, cells, life, stress, etc.)
            for ds_name in base_grp.keys():
                base_data = base_grp[ds_name][:]
                out_grp.create_dataset(ds_name, data=base_data)
                
            # 3. Calculate NEW exact geometric features
            mesh_pos = base_grp['mesh_pos'][:]
            cells = base_grp['cells'][:]
            
            try:
                # --- NEW ADDITION START ---
                # Unpack the 4th returned variable: dist_to_step
                normals, curvature, is_surface, dist_to_step = compute_exact_geometry(mesh_pos, cells, step_z_location)
                # --- NEW ADDITION END ---
                
                # 4. Save new features to the enriched H5
                out_grp.create_dataset("normals", data=normals)
                out_grp.create_dataset("curvature", data=curvature)
                out_grp.create_dataset("is_surface", data=is_surface)
                # --- NEW ADDITION START ---
                out_grp.create_dataset("dist_to_step", data=dist_to_step)
                # --- NEW ADDITION END ---
                
            except Exception as e:
                print(f"  [Error] Failed to compute exact geometry for {group_name}: {e}")
                continue
                
            if (idx + 1) % 10 == 0 or (idx + 1) == total_groups:
                print(f"Processed {idx + 1}/{total_groups} meshes...")
                
    elapsed = time.time() - start_time
    print(f"\nSuccessfully enriched {total_groups} datasets in {elapsed:.2f} seconds.")
    print(f"Enriched file saved to: {output_h5_path}")


if __name__ == "__main__":
    # --- VERIFY THESE PATHS MATCH YOUR SYSTEM ---
    BASE_H5 = r"/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/step_shaft_tensile_torsion.h5"
    # Note: I changed the output name slightly so you don't overwrite your 8-feature backup!
    ENRICHED_H5 = r"/home/coldforging/cold_forging_2D_github/project_fatigue/_DATA/step_shaft_tensile_torsion_enriched9.h5"

    # --- NEW ADDITION START ---
    STEP_Z = 20
    enrich_h5_dataset(BASE_H5, ENRICHED_H5, step_z_location=STEP_Z)
    # --- NEW ADDITION END ---
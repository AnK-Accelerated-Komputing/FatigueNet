#9 features .......................................................................................................

import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import random

# Threshold is still here for legacy modes, but ignored by 'full_regressor'
FATIGUE_LIFE_THRESHOLD = 10**5 

class TrajectoryDataset(Dataset):
    
    #Modes:
    #1. 'full_regressor': Trains on ALL data (Log10 Life). No masking.
    #2. 'classifier': Classifies LCF vs HCF.
    #3. 'lcf_regressor' / 'hcf_regressor': Filtered training.
    #4. 'full_eval': Raw data for final evaluation.
    
    def __init__(self, data_path, split='train', mode='full_regressor'):
        self.data_path = data_path
        self.split = split
        self.mode = mode 
        self.is_processed = self._is_processed()

        if not self.is_processed:
            self.process_data()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _is_processed(self):
        return os.path.exists(self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt'))

    def process_data(self):
        print(f"Processing data for mode: {self.mode}...")
        final_data = []
        
        with h5py.File(self.data_path, 'r') as h5_file:
            # 1. GET KEYS
            group_names = list(h5_file.keys())
            
            # 2. SORT THEM (CRITICAL FIX)
            # This forces a deterministic order (Alphabetical: 0, 1, 10, 100, 101...)
            # BEFORE the shuffle happens. This guarantees the starting state is identical
            # on every computer, every time.
            group_names.sort() 
            
            # 3. Reproducible shuffle
            random.seed(42)
            random.shuffle(group_names)
            
            total_groups = len(group_names)
            
            # Split logic (80/20)
            if self.split == 'train':
                selected_groups = group_names[:round(total_groups * 0.8)]
            elif self.split == 'val':
                selected_groups = group_names[round(total_groups * 0.8):]
            
            print(f"Selected {len(selected_groups)} groups for {self.split}")
            
            for group_name in selected_groups:
                try:
                    group = h5_file[group_name]
                    
                    # --- LOAD BASE DATA ---
                    cells = torch.tensor(group['cells'][:], dtype=torch.long)
                    mesh_pos = torch.tensor(group['mesh_pos'][:], dtype=torch.float32)
                    
                    # --- LOAD NEW GEOMETRIC FEATURES ---
                    normals = torch.tensor(group['normals'][:], dtype=torch.float32)
                    curvature = torch.tensor(group['curvature'][:], dtype=torch.float32)
                    is_surface = torch.tensor(group['is_surface'][:], dtype=torch.float32)
                    dist_to_step = torch.tensor(group['dist_to_step'][:], dtype=torch.float32)
                    
                    # --- LOAD FATIGUE LIFE ---
                    fatigue_life_raw = torch.tensor(group['fatigue_life'][:], dtype=torch.float32)
                    # Convert to Log10 for training stability
                    fatigue_life_log = torch.log10(fatigue_life_raw.clamp(min=1e-3))

                    # Ensure correct dimensions
                    if fatigue_life_log.dim() == 1:
                        fatigue_life_log = fatigue_life_log.unsqueeze(-1)

                    # --- MODE SELECTION ---
                    
                    # 1. FULL REGRESSOR (No Thresholds)
                    if self.mode == 'full_regressor':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log, 
                            'num_nodes': mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)

                    # 2. FULL EVAL (Raw Life for Evaluation)
                    elif self.mode == 'full_eval':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log, 
                            'fatigue_life_raw': fatigue_life_raw # Keep raw for real error calc
                        }
                        final_data.append(sample)

                    # 3. CLASSIFIER
                    elif self.mode == 'classifier':
                        target = (fatigue_life_raw > FATIGUE_LIFE_THRESHOLD).float().unsqueeze(-1)
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log,
                            'fatigue_class': target,
                        }
                        final_data.append(sample)

                    # 4. LCF/HCF (Filtered)
                    else: 
                        if self.mode == 'lcf_regressor':
                            mask = fatigue_life_raw <= FATIGUE_LIFE_THRESHOLD
                        else: # hcf_regressor
                            mask = fatigue_life_raw > FATIGUE_LIFE_THRESHOLD
                        
                        if torch.sum(mask) == 0: continue 
                        
                        squeezed_mask = mask.squeeze()
                        
                        # Apply Mask to ALL node-level features
                        filtered_mesh_pos = mesh_pos[squeezed_mask]
                        filtered_normals = normals[squeezed_mask]          # Added
                        filtered_curvature = curvature[squeezed_mask]      # Added
                        filtered_is_surface = is_surface[squeezed_mask]    # Added
                        filtered_dist_to_step = dist_to_step[squeezed_mask] # Added
                        filtered_fatigue_life = fatigue_life_log[squeezed_mask] 

                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': filtered_mesh_pos,
                            'normals': filtered_normals,         # Added
                            'curvature': filtered_curvature,     # Added
                            'is_surface': filtered_is_surface,   # Added
                            'dist_to_step': filtered_dist_to_step, # Added
                            'fatigue_life': filtered_fatigue_life,
                            'num_nodes': filtered_mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)
                        
                except Exception as e:
                    print(f"Error processing {group_name}: {e}")
                    continue

        save_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        torch.save(final_data, save_path)
        print(f"Saved {len(final_data)} samples to {save_path}")

    def load_data(self):
        load_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        return torch.load(load_path)





"""
#8features ----------------------------------------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import random

# Threshold is still here for legacy modes, but ignored by 'full_regressor'
FATIGUE_LIFE_THRESHOLD = 10**5 

class TrajectoryDataset(Dataset):


    def __init__(self, data_path, split='train', mode='full_regressor'):
        self.data_path = data_path
        self.split = split
        self.mode = mode 
        self.is_processed = self._is_processed()

        if not self.is_processed:
            self.process_data()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _is_processed(self):
        return os.path.exists(self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt'))

    def process_data(self):
        print(f"Processing data for mode: {self.mode}...")
        final_data = []
        
        with h5py.File(self.data_path, 'r') as h5_file:
            # 1. GET KEYS
            group_names = list(h5_file.keys())
            
            # 2. SORT THEM (CRITICAL FIX)
            # This forces a deterministic order (Alphabetical: 0, 1, 10, 100, 101...)
            # BEFORE the shuffle happens. This guarantees the starting state is identical
            # on every computer, every time.
            group_names.sort() 
            
            # 3. Reproducible shuffle
            random.seed(42)
            random.shuffle(group_names)
            
            total_groups = len(group_names)
            
            # Split logic (80/20)
            if self.split == 'train':
                selected_groups = group_names[:round(total_groups * 0.8)]
            elif self.split == 'val':
                selected_groups = group_names[round(total_groups * 0.8):]
            
            print(f"Selected {len(selected_groups)} groups for {self.split}")
            
            for group_name in selected_groups:
                try:
                    group = h5_file[group_name]
                    
                    # --- LOAD BASE DATA ---
                    cells = torch.tensor(group['cells'][:], dtype=torch.long)
                    mesh_pos = torch.tensor(group['mesh_pos'][:], dtype=torch.float32)
                    
                    # --- LOAD NEW GEOMETRIC FEATURES ---
                    normals = torch.tensor(group['normals'][:], dtype=torch.float32)
                    curvature = torch.tensor(group['curvature'][:], dtype=torch.float32)
                    is_surface = torch.tensor(group['is_surface'][:], dtype=torch.float32)
                    #dist_to_step = torch.tensor(group['dist_to_step'][:], dtype=torch.float32)
                    
                    # --- LOAD FATIGUE LIFE ---
                    fatigue_life_raw = torch.tensor(group['fatigue_life'][:], dtype=torch.float32)
                    # Convert to Log10 for training stability
                    fatigue_life_log = torch.log10(fatigue_life_raw.clamp(min=1e-3))

                    # Ensure correct dimensions
                    if fatigue_life_log.dim() == 1:
                        fatigue_life_log = fatigue_life_log.unsqueeze(-1)

                    # --- MODE SELECTION ---
                    
                    # 1. FULL REGRESSOR (No Thresholds)
                    if self.mode == 'full_regressor':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            #'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log, 
                            'num_nodes': mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)

                    # 2. FULL EVAL (Raw Life for Evaluation)
                    elif self.mode == 'full_eval':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            #'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log, 
                            'fatigue_life_raw': fatigue_life_raw # Keep raw for real error calc
                        }
                        final_data.append(sample)

                    # 3. CLASSIFIER
                    elif self.mode == 'classifier':
                        target = (fatigue_life_raw > FATIGUE_LIFE_THRESHOLD).float().unsqueeze(-1)
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'normals': normals,          # Added
                            'curvature': curvature,      # Added
                            'is_surface': is_surface,    # Added
                            #'dist_to_step': dist_to_step, # Added
                            'fatigue_life': fatigue_life_log,
                            'fatigue_class': target,
                        }
                        final_data.append(sample)

                    # 4. LCF/HCF (Filtered)
                    else: 
                        if self.mode == 'lcf_regressor':
                            mask = fatigue_life_raw <= FATIGUE_LIFE_THRESHOLD
                        else: # hcf_regressor
                            mask = fatigue_life_raw > FATIGUE_LIFE_THRESHOLD
                        
                        if torch.sum(mask) == 0: continue 
                        
                        squeezed_mask = mask.squeeze()
                        
                        # Apply Mask to ALL node-level features
                        filtered_mesh_pos = mesh_pos[squeezed_mask]
                        filtered_normals = normals[squeezed_mask]          # Added
                        filtered_curvature = curvature[squeezed_mask]      # Added
                        filtered_is_surface = is_surface[squeezed_mask]    # Added
                        #filtered_dist_to_step = dist_to_step[squeezed_mask] # Added
                        filtered_fatigue_life = fatigue_life_log[squeezed_mask] 

                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': filtered_mesh_pos,
                            'normals': filtered_normals,         # Added
                            'curvature': filtered_curvature,     # Added
                            'is_surface': filtered_is_surface,   # Added
                            #'dist_to_step': filtered_dist_to_step, # Added
                            'fatigue_life': filtered_fatigue_life,
                            'num_nodes': filtered_mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)
                        
                except Exception as e:
                    print(f"Error processing {group_name}: {e}")
                    continue

        save_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        torch.save(final_data, save_path)
        print(f"Saved {len(final_data)} samples to {save_path}")

    def load_data(self):
        load_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        return torch.load(load_path)

"""




























#....................................................................................................................
# Mesh position only as input datsets#
#...................................................................................................................



"""
import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import random
# Threshold is still here for legacy modes, but ignored by 'full_regressor'
FATIGUE_LIFE_THRESHOLD = 10**5 

class TrajectoryDataset(Dataset):
    
    #Modes:
    #1. 'full_regressor': Trains on ALL data (Log10 Life). No masking.
    #2. 'classifier': Classifies LCF vs HCF.
    #3. 'lcf_regressor' / 'hcf_regressor': Filtered training.
    #4. 'full_eval': Raw data for final evaluation.
    
    def __init__(self, data_path, split='train', mode='full_regressor'):
        self.data_path = data_path
        self.split = split
        self.mode = mode 
        self.is_processed = self._is_processed()

        if not self.is_processed:
            self.process_data()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _is_processed(self):
        return os.path.exists(self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt'))

    def process_data(self):
        print(f"Processing data for mode: {self.mode}...")
        final_data = []
        
        with h5py.File(self.data_path, 'r') as h5_file:
            # 1. GET KEYS
            group_names = list(h5_file.keys())
            
            # 2. SORT THEM (CRITICAL FIX)
            # This forces a deterministic order (Alphabetical: 0, 1, 10, 100, 101...)
            # BEFORE the shuffle happens. This guarantees the starting state is identical
            # on every computer, every time.
            group_names.sort() 
            
            # 3. Reproducible shuffle
            random.seed(42)
            random.shuffle(group_names)
            
            total_groups = len(group_names)
            
            # Split logic (80/20)
            if self.split == 'train':
                selected_groups = group_names[:round(total_groups * 0.8)]
            elif self.split == 'val':
                selected_groups = group_names[round(total_groups * 0.8):]
            
            print(f"Selected {len(selected_groups)} groups for {self.split}")
            
            for group_name in selected_groups:
                try:
                    group = h5_file[group_name]
                    
                    # Load Data
                    cells = torch.tensor(group['cells'][:], dtype=torch.long)
                    mesh_pos = torch.tensor(group['mesh_pos'][:], dtype=torch.float32)
                    
                    fatigue_life_raw = torch.tensor(group['fatigue_life'][:], dtype=torch.float32)
                    # Convert to Log10 for training stability
                    fatigue_life_log = torch.log10(fatigue_life_raw.clamp(min=1e-3))

                    # Ensure correct dimensions
                    if fatigue_life_log.dim() == 1:
                        fatigue_life_log = fatigue_life_log.unsqueeze(-1)

                    # --- MODE SELECTION ---
                    
                    # 1. FULL REGRESSOR (No Thresholds)
                    if self.mode == 'full_regressor':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'fatigue_life': fatigue_life_log, # Training on Log10 life
                            'num_nodes': mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)

                    # 2. FULL EVAL (Raw Life for Evaluation)
                    elif self.mode == 'full_eval':
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'fatigue_life': fatigue_life_log, 
                            'fatigue_life_raw': fatigue_life_raw # Keep raw for real error calc
                        }
                        final_data.append(sample)

                    # 3. CLASSIFIER
                    elif self.mode == 'classifier':
                        target = (fatigue_life_raw > FATIGUE_LIFE_THRESHOLD).float().unsqueeze(-1)
                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': mesh_pos,
                            'fatigue_life': fatigue_life_log,
                            'fatigue_class': target,
                        }
                        final_data.append(sample)

                    # 4. LCF/HCF (Filtered)
                    else: 
                        if self.mode == 'lcf_regressor':
                            mask = fatigue_life_raw <= FATIGUE_LIFE_THRESHOLD
                        else: # hcf_regressor
                            mask = fatigue_life_raw > FATIGUE_LIFE_THRESHOLD
                        
                        if torch.sum(mask) == 0: continue 
                        
                        squeezed_mask = mask.squeeze()
                        
                        # Apply Mask
                        filtered_mesh_pos = mesh_pos[squeezed_mask]
                        filtered_fatigue_life = fatigue_life_log[squeezed_mask] 

                        sample = {
                            'group_id': group_name,
                            'cells': cells,
                            'mesh_pos': filtered_mesh_pos,
                            'fatigue_life': filtered_fatigue_life,
                            'num_nodes': filtered_mesh_pos.shape[0],
                            'num_cells': cells.shape[0]
                        }
                        final_data.append(sample)
                        
                except Exception as e:
                    print(f"Error processing {group_name}: {e}")
                    continue

        save_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        torch.save(final_data, save_path)
        print(f"Saved {len(final_data)} samples to {save_path}")

    def load_data(self):
        load_path = self.data_path.replace('.h5', f'_{self.split}_{self.mode}.pt')
        return torch.load(load_path)

"""
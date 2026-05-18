import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, split='train', normalize=True):
        self.data_path = data_path
        self.split = split
        self.normalize = normalize
        self.is_processed = self._is_processed()

        if not self.is_processed:
            self.process_data()
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _is_processed(self):
        return os.path.exists(self.data_path.replace('.h5', f'_{self.split}.pt'))

    def process_data(self):
        print("Processing shaft fatigue data...")
        
        final_data = []
        
        with h5py.File(self.data_path, 'r') as h5_file:
            # Get all group names (group_0 to group_15)
            group_names = [f'group_{i}' for i in range(16)]  # Groups group_0 to group_15
            total_groups = len(group_names)
            
            # Split data: 70% train, 15% val, 15% test
            if self.split == 'train':
                group_limit = round(total_groups * 0.7)  # 0 to 10 (11 groups)
                selected_groups = group_names[:group_limit]
            elif self.split == 'val':
                group_limit_start = round(total_groups * 0.7)  # 11
                group_limit_end = round(total_groups * 0.85)   # 13
                if group_limit_end == group_limit_start:
                    selected_groups = group_names[group_limit_start:]
                else:
                    selected_groups = group_names[group_limit_start:group_limit_end]
            else:  # test
                group_limit_start = round(total_groups * 0.85)  # 14
                if group_limit_start == total_groups:
                    group_limit_start -= 1
                selected_groups = group_names[group_limit_start:]
            
            print(f"Selected groups for {self.split}: {selected_groups}")
            
            # Store normalization statistics for fatigue life
            all_fatigue_life = []
            
            # First pass: collect all fatigue life data for normalization
            if self.normalize:
                for group_name in group_names:
                    if group_name in h5_file:
                        group = h5_file[group_name]
                        if 'fatigue_life' in group:
                            fatigue_life = group['fatigue_life'][:]
                            all_fatigue_life.append(fatigue_life)
                
                if all_fatigue_life:
                    all_fatigue_life = np.concatenate(all_fatigue_life)
                    self.fatigue_mean = np.mean(all_fatigue_life)
                    self.fatigue_std = np.std(all_fatigue_life)
                    print(f"Fatigue life normalization - Mean: {self.fatigue_mean:.4f}, Std: {self.fatigue_std:.4f}")
            
            # Second pass: process selected groups
            for group_name in selected_groups:
                if group_name not in h5_file:
                    print(f"Warning: Group {group_name} not found in file")
                    continue
                    
                print(f"Processing group {group_name}")
                group = h5_file[group_name]
                
                # Extract data from each group
                try:
                    # Read datasets
                    cells = torch.tensor(group['cells'][:], dtype=torch.long)  # Shape: (num_cells, 4)
                    mesh_pos = torch.tensor(group['mesh_pos'][:], dtype=torch.float32)  # Shape: (num_nodes, 3)
                    node_type = torch.tensor(group['node_type'][:], dtype=torch.long)  # Shape: (num_nodes,) or (num_nodes, 1)
                    fatigue_life = torch.tensor(group['fatigue_life'][:], dtype=torch.float32)  # Shape: (num_nodes,)
                    
                    # Ensure node_type is 2D
                    if node_type.dim() == 1:
                        node_type = node_type.unsqueeze(-1)
                    
                    # Normalize fatigue life if requested
                    if self.normalize and hasattr(self, 'fatigue_std') and self.fatigue_std > 0:
                        fatigue_life = (fatigue_life - self.fatigue_mean) / self.fatigue_std
                    
                    # Create data sample
                    sample = {
                        'group_id': int(group_name.split('_')[1]),  # Extract number from group_X
                        'cells': cells,                    # (num_cells, 4)
                        'mesh_pos': mesh_pos,              # (num_nodes, 3)
                        'node_type': node_type,            # (num_nodes, 1)
                        'fatigue_life': fatigue_life,      # (num_nodes,)
                        'num_nodes': mesh_pos.shape[0],
                        'num_cells': cells.shape[0]
                    }
                    
                    final_data.append(sample)
                    
                except Exception as e:
                    print(f"Error processing group {group_name}: {e}")
                    continue

        # Save processed data
        save_path = self.data_path.replace('.h5', f'_{self.split}.pt')
        torch.save(final_data, save_path)
        
        # Save normalization statistics
        if self.normalize and hasattr(self, 'fatigue_mean'):
            stats_path = self.data_path.replace('.h5', '_normalization_stats.pt')
            torch.save({
                'fatigue_mean': self.fatigue_mean,
                'fatigue_std': self.fatigue_std
            }, stats_path)
        
        print(f"Data successfully saved to {save_path}")
        print(f"Total samples in {self.split}: {len(final_data)}")

    def load_data(self):
        return torch.load(self.data_path.replace('.h5', f'_{self.split}.pt'))
    
    def get_normalization_stats(self):
        """Get normalization statistics for fatigue life"""
        if self.normalize:
            stats_path = self.data_path.replace('.h5', '_normalization_stats.pt')
            if os.path.exists(stats_path):
                return torch.load(stats_path)
        return None
    
    def denormalize_fatigue_life(self, normalized_fatigue_life):
        """Convert normalized fatigue life back to original scale"""
        stats = self.get_normalization_stats()
        if stats is not None:
            return normalized_fatigue_life * stats['fatigue_std'] + stats['fatigue_mean']
        return normalized_fatigue_life


def explore_h5_structure(data_path):
    """Utility function to explore the structure of your H5 file"""
    print(f"Exploring structure of {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        
        # Check first few groups
        available_groups = [key for key in f.keys() if key.startswith('group_')]
        for i in range(min(3, len(available_groups))):
            group_name = available_groups[i]
            if group_name in f:
                group = f[group_name]
                print(f"\nGroup {group_name}:")
                print(f"  Keys: {list(group.keys())}")
                
                # Check shapes of each dataset
                for key in group.keys():
                    try:
                        shape = group[key].shape
                        dtype = group[key].dtype
                        print(f"  {key}: shape={shape}, dtype={dtype}")
                    except:
                        print(f"  {key}: Could not read shape/dtype")


def main():
    data_file = r"D:\AnK\Project Fatigue\Notch shaft\fatigue_net\PressNet\datasets\extracted_data\shaft_.h5"
    
    # First, explore the structure to understand the actual group names
    print("=== EXPLORING H5 FILE STRUCTURE ===")
    explore_h5_structure(data_file)
    print("=" * 50)
    
    # Now create datasets with correct group names
    print("Creating training dataset...")
    train_dataset = TrajectoryDataset(data_file, split='train', normalize=True)
    
    print("\nCreating validation dataset...")
    val_dataset = TrajectoryDataset(data_file, split='val', normalize=True)
    
    print("\nCreating test dataset...")
    test_dataset = TrajectoryDataset(data_file, split='test', normalize=True)
    
    # Create dataloaders only if datasets are not empty
    if len(train_dataset) > 0:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    if len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
    if len(test_dataset) > 0:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Examine first batch if training data exists
    if len(train_dataset) > 0:
        print("\nFirst training batch:")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
            break
    
    # Check normalization stats
    stats = train_dataset.get_normalization_stats()
    if stats:
        print(f"\nNormalization stats:")
        print(f"  Fatigue life mean: {stats['fatigue_mean']:.4f}")
        print(f"  Fatigue life std: {stats['fatigue_std']:.4f}")


if __name__ == '__main__':
    main()
    
    """The .pt file is saved automatically inside the TrajectoryDataset class — in the constructor 
    (__init__) — via the call to self.process_data() if the data hasn't been processed yet.
    You can load it later using the load_data() method.
    This will load the processed data from the .pt file, which contains all the necessary tensors.
    def __init__(self, data_path, split='train', normalize=True):
    self.data_path = data_path
    self.split = split
    self.normalize = normalize
    self.is_processed = self._is_processed()

    if not self.is_processed:
        self.process_data()     # <- THIS saves the .pt file
    self.data = self.load_data()


So when you do:

train_dataset = TrajectoryDataset(data_file, split='train', normalize=True)
Here’s what happens step-by-step:

_is_processed() checks whether shaft__train.pt exists.

If it doesn't, process_data() is called:

It loads the HDF5 groups.

Prepares the data.

Saves it to shaft__train.pt via:

torch.save(final_data, save_path)
The saved .pt file is then loaded by self.load_data().

Output Files Created:
If your original file is named:

plaintext
shaft_.h5


-------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
def load_data(self):
    return torch.load(self.data_path.replace('.h5', f'_{self.split}.pt'))
    This method loads the processed data from the .pt file created during processing.
    💡 What It Does:
This method loads the preprocessed .pt file (a PyTorch object file) for the given split (train, val, or test) 
and returns it as a Python object 


— specifically, a list of dictionaries (your processed data samples).
What is in shaft__train.pt?
In your process_data() method, you saved a list of samples like this:

sample = {
    'group_id': int(group_name.split('_')[1]),
    'cells': cells,                # Tensor of shape (num_cells, 4)
    'mesh_pos': mesh_pos,          # Tensor of shape (num_nodes, 3)
    'node_type': node_type,        # Tensor of shape (num_nodes, 1)
    'fatigue_life': fatigue_life,  # Tensor of shape (num_nodes,)
    'num_nodes': mesh_pos.shape[0],
    'num_cells': cells.shape[0]
}

And then saved all of them in a list:
final_data.append(sample)
torch.save(final_data, save_path)
So load_data() simply loads this list of dictionaries back into memory.

After loading:

self.data = load_data()
You now have:
self.data = [
    { 'group_id': ..., 'cells': ..., 'mesh_pos': ..., ... },
    { 'group_id': ..., 'cells': ..., 'mesh_pos': ..., ... },
    ...
]


"""
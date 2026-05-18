#....................................................................................................................
#9 features datasets
#...................................................................................................................


import torch
from torch import nn as nn
from models import normalization
from models import regDGCNN_seg

class Model(nn.Module):
    def __init__(self, params, core_model_name="regDGCNN_seg"):                                        
        super(Model, self).__init__() 
        self._params = params
        self.purpose = self._params.get('purpose', 'regressor') 
        self.k = self._params.get('k', 10)
        
        self.core_model_name = core_model_name
        self._model_type = core_model_name 
        
        # Initialize normalizer ONLY for regressors
        if 'regressor' in self.purpose:
            self._output_life_normalizer = normalization.Normalizer(size=1, name='output_life_normalizer')
        else:
            self._output_life_normalizer = None

        # Safely get input_size, defaulting to 9
        input_size = params.get('input_size', 9)
        print(f"Model initialized for '{self.purpose}' with {input_size} inputs.")

        if core_model_name == 'regDGCNN_seg':
            self.learned_model = regDGCNN_seg.regDGCNN_seg(   
                output_size=params.get('output_size', 1),            
                input_dims=input_size, 
                k=self.k, 
                emb_dims=1024,  
                dropout=0.1 
            )
        else:
            raise ValueError(f"Unsupported core model: {self.core_model_name}")

    def forward(self, inputs, is_training):
        processed_features = self.process_inputs(inputs)
        raw_network_output = self.learned_model(processed_features)
        
        # 1. If we are training, we ALWAYS need the normalized raw output for the loss function.
        # 2. Classifiers do not use normalizers, so they also return raw logits.
        if is_training or 'classifier' in self.purpose:
            return raw_network_output
            
        # 3. If we are validating/evaluating a Regressor, denormalize automatically!
        else:
            return self._update(raw_network_output)

    def _update(self, per_node_network_output):
        """Only used for final evaluation/inference on regressors."""
        if self._output_life_normalizer is None:
            raise RuntimeError("Normalizer not initialized for this model.")
        fatigue_pred = self._output_life_normalizer.inverse(per_node_network_output[:, :]) 
        return fatigue_pred

    def process_inputs(self, inputs): 
        """
        Processes input tensors, selectively normalizes them correctly, 
        concatenates them, and prepares features for the DGCNN.
        """
        device = next(self.parameters()).device
        pos = inputs['mesh_pos'].to(device)

        has_batch = pos.dim() == 3

        # 1. Normalize Coordinates (Zero mean, Unit variance)
        if not has_batch: # Shape: (N, 3)
            pos_norm = (pos - pos.mean(dim=0)) / (pos.std(dim=0) + 1e-8)
        else:              # Shape: (Batch, N, 3)
            pos_norm = (pos - pos.mean(dim=1, keepdim=True)) / (pos.std(dim=1, keepdim=True) + 1e-8)

        # Start a list of features to concatenate
        features_list = [pos_norm]

        # 2. Add Normals (DO NOT NORMALIZE)
        if 'normals' in inputs:
            features_list.append(inputs['normals'].to(device))

        # 3. Add Curvature (NORMALIZE)
        if 'curvature' in inputs:
            curv = inputs['curvature'].to(device)
            if not has_batch:
                curv_norm = (curv - curv.mean(dim=0)) / (curv.std(dim=0) + 1e-8)
            else:
                curv_norm = (curv - curv.mean(dim=1, keepdim=True)) / (curv.std(dim=1, keepdim=True) + 1e-8)
            features_list.append(curv_norm)

        # --- NEW: 4. Add Distance to Step (NORMALIZE) ---
        if 'dist_to_step' in inputs:
            dist = inputs['dist_to_step'].to(device)
            if not has_batch:
                dist_norm = (dist - dist.mean(dim=0)) / (dist.std(dim=0) + 1e-8)
            else:
                dist_norm = (dist - dist.mean(dim=1, keepdim=True)) / (dist.std(dim=1, keepdim=True) + 1e-8)
            features_list.append(dist_norm)

        # 5. Add Surface Mask (DO NOT NORMALIZE)
        if 'is_surface' in inputs:
            features_list.append(inputs['is_surface'].to(device))

        # 6. Concatenate everything into one tensor -> Shape: (N, Features) or (B, N, Features)
        combined_features = torch.cat(features_list, dim=-1)

        # 7. Ensure (Batch, Channels, N) format for DGCNN
        if combined_features.dim() == 2: # If (N, C)
            x = combined_features.permute(1, 0).unsqueeze(0) 
        elif combined_features.dim() == 3: # If (Batch, N, C)
            x = combined_features.permute(0, 2, 1)
            
        return x 
     
    def fit_normalizer(self, dataloader):
        if 'regressor' not in self.purpose: 
            return

        print("Fitting output life normalizer...")
        all_vals = []
        for data in dataloader:
            if 'fatigue_life' in data:
                all_vals.append(data['fatigue_life'].view(-1))
        
        all_vals = torch.cat(all_vals)
        mean = torch.mean(all_vals)
        std = torch.std(all_vals)

        self._output_life_normalizer.set_stats(mean, std)
        print(f"Stats set: Mean={mean.item():.4f}, Std={std.item():.4f}")

    def get_output_life_normalizer(self):
        return self._output_life_normalizer

    def save_model(self, path):
        torch.save(self.learned_model.state_dict(), path + "_learned_model.pth")
        if 'regressor' in self.purpose and self._output_life_normalizer is not None:
            torch.save(self._output_life_normalizer, path + "_output_life_normalizer.pth")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.learned_model.load_state_dict(torch.load(path + "_learned_model.pth", map_location=next(self.parameters()).device))
        if 'regressor' in self.purpose:
            try:
                self._output_life_normalizer = torch.load(path + "_output_life_normalizer.pth")
                print("Loaded model + normalizer.")
            except:
                print("Warning: Normalizer file not found.")


#....................................................................................................................
#8 features datasets
#...................................................................................................................




'''
import torch
from torch import nn as nn
from models import normalization
from models import regDGCNN_seg

class Model(nn.Module):
    def __init__(self, params, core_model_name="regDGCNN_seg"):                                        
        super(Model, self).__init__() 
        self._params = params
        self.purpose = self._params.get('purpose', 'regressor') 
        self.k = self._params['k']
        
        self.core_model_name = core_model_name
        self._model_type = core_model_name 
        
        # Initialize normalizer ONLY for regressors
        if 'regressor' in self.purpose:
            self._output_life_normalizer = normalization.Normalizer(size=1, name='output_life_normalizer')
        else:
            self._output_life_normalizer = None

        print(f"Model initialized for '{self.purpose}' with {params['input_size']} inputs.")

        if core_model_name == 'regDGCNN_seg':
            self.learned_model = regDGCNN_seg.regDGCNN_seg(   
                output_size=params['output_size'],            
                input_dims=params['input_size'],  # This will be 8 for your new features
                k=self.k, 
                emb_dims=1024,  
                dropout=0.1 
            )
        else:
            raise ValueError(f"Unsupported core model: {self.core_model_name}")

    def forward(self, inputs, is_training):
        processed_features = self.process_inputs(inputs)
        raw_network_output = self.learned_model(processed_features)
        
        # 1. If we are training, we ALWAYS need the normalized raw output for the loss function.
        # 2. Classifiers do not use normalizers, so they also return raw logits.
        if is_training or 'classifier' in self.purpose:
            return raw_network_output
            
        # 3. If we are validating/evaluating a Regressor, denormalize automatically!
        else:
            return self._update(raw_network_output)

    def _update(self, per_node_network_output):
        """Only used for final evaluation/inference on regressors."""
        if self._output_life_normalizer is None:
            raise RuntimeError("Normalizer not initialized for this model.")
        fatigue_pred = self._output_life_normalizer.inverse(per_node_network_output[:, :]) 
        return fatigue_pred

    def process_inputs(self, inputs): 
        """
        Processes input tensors, selectively normalizes them correctly, 
        concatenates them, and prepares features for the DGCNN.
        """
        device = next(self.parameters()).device
        pos = inputs['mesh_pos'].to(device)

        # 1. Normalize Coordinates (Zero mean, Unit variance)
        if pos.dim() == 2: # Shape: (N, 3)
            pos_norm = (pos - pos.mean(dim=0)) / (pos.std(dim=0) + 1e-8)
        else:              # Shape: (Batch, N, 3)
            pos_norm = (pos - pos.mean(dim=1, keepdim=True)) / (pos.std(dim=1, keepdim=True) + 1e-8)

        # Start a list of features to concatenate
        features_list = [pos_norm]

        # 2. Add Normals (DO NOT NORMALIZE)
        if 'normals' in inputs:
            features_list.append(inputs['normals'].to(device))

        # 3. Add Curvature (NORMALIZE)
        if 'curvature' in inputs:
            curv = inputs['curvature'].to(device)
            if curv.dim() == 2:
                curv_norm = (curv - curv.mean(dim=0)) / (curv.std(dim=0) + 1e-8)
            else:
                curv_norm = (curv - curv.mean(dim=1, keepdim=True)) / (curv.std(dim=1, keepdim=True) + 1e-8)
            features_list.append(curv_norm)

        # 4. Add Surface Mask (DO NOT NORMALIZE)
        if 'is_surface' in inputs:
            features_list.append(inputs['is_surface'].to(device))

        # 5. Concatenate everything into one tensor -> Shape: (N, Features) or (B, N, Features)
        combined_features = torch.cat(features_list, dim=-1)

        # 6. Ensure (Batch, Channels, N) format for DGCNN
        if combined_features.dim() == 2: # If (N, C)
            x = combined_features.permute(1, 0).unsqueeze(0) 
        elif combined_features.dim() == 3: # If (Batch, N, C)
            x = combined_features.permute(0, 2, 1)
            
        return x 
     
    def fit_normalizer(self, dataloader):
        if 'regressor' not in self.purpose: 
            return

        print("Fitting output life normalizer...")
        all_vals = []
        for data in dataloader:
            if 'fatigue_life' in data:
                all_vals.append(data['fatigue_life'].view(-1))
        
        all_vals = torch.cat(all_vals)
        mean = torch.mean(all_vals)
        std = torch.std(all_vals)

        self._output_life_normalizer.set_stats(mean, std)
        print(f"Stats set: Mean={mean.item():.4f}, Std={std.item():.4f}")

    def get_output_life_normalizer(self):
        return self._output_life_normalizer

    def save_model(self, path):
        torch.save(self.learned_model.state_dict(), path + "_learned_model.pth")
        if 'regressor' in self.purpose and self._output_life_normalizer is not None:
            torch.save(self._output_life_normalizer, path + "_output_life_normalizer.pth")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.learned_model.load_state_dict(torch.load(path + "_learned_model.pth", map_location=next(self.parameters()).device))
        if 'regressor' in self.purpose:
            try:
                self._output_life_normalizer = torch.load(path + "_output_life_normalizer.pth")
                print("Loaded model + normalizer.")
            except:
                print("Warning: Normalizer file not found.")

'''

















#....................................................................................................................
# Mesh position only as input datsets#
#...................................................................................................................

"""

import torch
from torch import nn as nn
from models import normalization
from models import regDGCNN_seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, params, core_model_name="regDGCNN_seg"):
        super(Model, self).__init__()
        self._params = params
        self.purpose = self._params.get('purpose', 'regressor') 
        self.k = self._params['k']
        
        # FIX: Use core_model_name directly
        self._model_type = core_model_name 
        self.core_model_name = core_model_name
        
        # Initialize normalizer for ANY regressor (including full_regressor)
        if 'regressor' in self.purpose:
            self._output_life_normalizer = normalization.Normalizer(size=1, name='output_life_normalizer')
        else:
            self._output_life_normalizer = None
 
        print(f"Model initialized for '{self.purpose}' with {params['input_size']} inputs.")

        if core_model_name == 'regDGCNN_seg':
            self.learned_model = regDGCNN_seg.regDGCNN_seg(
                output_size=self._params['output_size'],
                input_dims=params['input_size'],
                k=self.k, 
                emb_dims=1024,  
                dropout=0.1 
            )
        else:
            raise ValueError(f"Unsupported model: {self.core_model_name}")

    def forward(self, inputs, is_training):
        processed_features = self.process_inputs(inputs)
        raw_network_output = self.learned_model(processed_features)

        if is_training:
            return raw_network_output
        else:
            if 'regressor' in self.purpose:
                return self._update(raw_network_output)
            else: 
                return raw_network_output

    def _update(self, per_node_network_output):
        if self._output_life_normalizer is None:
            raise RuntimeError("Normalizer not initialized.")
        return self._output_life_normalizer.inverse(per_node_network_output)

    def process_inputs(self, inputs):
        
        #FIX: Only extract mesh_pos (x,y,z) to avoid crashing on missing data.
        
        device = next(self.parameters()).device
        mesh_pos = inputs['mesh_pos'].to(device) # Shape: (Batch, N, 3)

        # Ensure (Batch, 3, N) format for DGCNN
        if mesh_pos.dim() == 2: # If (N, 3)
            x = mesh_pos.permute(1, 0).unsqueeze(0) 
        elif mesh_pos.dim() == 3: # If (B, N, 3)
            x = mesh_pos.permute(0, 2, 1)
            
        return x 
    
    def fit_normalizer(self, dataloader):
        if 'regressor' not in self.purpose: return

        print("Fitting normalizer...")
        all_vals = []
        for data in dataloader:
            if 'fatigue_life' in data:
                all_vals.append(data['fatigue_life'].view(-1))
        
        all_vals = torch.cat(all_vals)
        mean = torch.mean(all_vals)
        std = torch.std(all_vals)

        self._output_life_normalizer.set_stats(mean, std)
        print(f"Stats set: Mean={mean.item():.4f}, Std={std.item():.4f}")

    def get_output_life_normalizer(self):
        return self._output_life_normalizer
    
    def save_model(self, path):
        torch.save(self.learned_model.state_dict(), path + "_learned_model.pth")
        if 'regressor' in self.purpose and self._output_life_normalizer is not None:
            torch.save(self._output_life_normalizer, path + "_output_life_normalizer.pth")
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.learned_model.load_state_dict(torch.load(path + "_learned_model.pth", map_location=next(self.parameters()).device))
        if 'regressor' in self.purpose:
            try:
                self._output_life_normalizer = torch.load(path + "_output_life_normalizer.pth")
                print("Loaded model + normalizer.")
            except:
                print("Warning: Normalizer file not found.")

"""
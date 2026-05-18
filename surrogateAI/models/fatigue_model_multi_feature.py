import torch
from torch import nn as nn
from models import normalization
from models import regDGCNN_seg

class Model(nn.Module):
    def __init__(self, params, core_model_name="regDGCNN_seg"):                                        
        super(Model, self).__init__() 
        self._params = params
        
        # Output normalizer
        self._output_life_normalizer = normalization.Normalizer(size=1, name='output_pos_normalizer')
        self.k = params['k']                                                                           
        self.core_model_name = core_model_name

        if core_model_name == 'regDGCNN_seg':
            self.core_model = regDGCNN_seg
            self.is_multigraph = False

            self.learned_model = regDGCNN_seg.regDGCNN_seg(   
                output_size=params['output_size'],            
                input_dims=params['input_size'],  # Make sure this is 8 in your train.py args!
                k=self.k, 
                emb_dims=1024,  
                dropout=0.1 
            )
        else:
            raise ValueError(f"Unsupported core model: {self.core_model_name}")

    def forward(self, inputs, is_training):
        # We always return the raw network output during training/validation 
        # so the loss functions in train.py work correctly!
        processed_features = self.process_inputs(inputs)
        return self.learned_model(processed_features)

    def _update(self, per_node_network_output):
        """Only used for final evaluation/inference on new CAD parts."""
        fatigue_pred = self._output_life_normalizer.inverse(per_node_network_output[:, :]) 
        return fatigue_pred

    def process_inputs(self, inputs): 
        """
        Processes input tensors, normalizes them correctly, 
        and prepares features for the DGCNN.
        """
        device = next(self.parameters()).device
        
        # 1. Load the new physical features
        mesh_pos = inputs['mesh_pos'].to(device)       # (N, 3)
        normals = inputs['normals'].to(device)         # (N, 3)
        curvature = inputs['curvature'].to(device)     # (N, 1)
        is_surface = inputs['is_surface'].to(device)   # (N, 1)
        
        # 2. CRITICAL: Normalize coordinates and curvature
        # (Subtract mean, divide by std)
        pos_norm = (mesh_pos - mesh_pos.mean(dim=0)) / (mesh_pos.std(dim=0) + 1e-8)
        curv_norm = (curvature - curvature.mean(dim=0)) / (curvature.std(dim=0) + 1e-8)

        # 3. Concatenate all features
        # Note: normals and is_surface are NOT normalized!
        x = torch.cat([
            pos_norm,    # (N, 3)
            normals,     # (N, 3)
            curv_norm,   # (N, 1)
            is_surface   # (N, 1)
        ], dim=1)        # Total = (N, 8)
        
        # 4. Reshape for DGCNN expectations
        x = x.T.unsqueeze(0)  # Shape becomes: [1, 8, N]
        
        return x 
     
    def get_output_life_normalizer(self):
        return self._output_life_normalizer
    
    def fit_normalizer(self, dataloader):
        print("Fitting output normalizer...")
        all_vals = []
        for data in dataloader:
            if 'fatigue_life' in data:
                all_vals.append(data['fatigue_life'].view(-1))
        
        all_vals = torch.cat(all_vals)
        mean = torch.mean(all_vals)
        std = torch.std(all_vals)

        self._output_life_normalizer.set_stats(mean, std)
        print(f"Stats set: Mean={mean.item():.4f}, Std={std.item():.4f}")

    def save_model(self, path):
        torch.save(self.learned_model.state_dict(), path + "_learned_model.pth")
        torch.save(self._output_life_normalizer, path + "_output_life_normalizer.pth")

    def load_model(self, path):
        self.learned_model.load_state_dict(torch.load(path + "_learned_model.pth", map_location=next(self.parameters()).device))
        self._output_life_normalizer = torch.load(path + "_output_life_normalizer.pth")
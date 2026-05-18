import torch
from torch import nn as nn
from models import normalization
from models import transolver

class Model(nn.Module):
    def __init__(self, params, core_model_name="transolver"):                                        
        super(Model, self).__init__() 
        self._params = params
        self.purpose = self._params.get('purpose', 'regressor') 
        self.core_model_name = core_model_name
        
        # Initialize normalizer ONLY for regressors
        if 'regressor' in self.purpose:
            self._output_life_normalizer = normalization.Normalizer(size=1, name='output_life_normalizer')
        else:
            self._output_life_normalizer = None

        print(f"Model initialized for '{self.purpose}' with {params.get('input_size', 8)} inputs.")

        if core_model_name == 'transolver':
            # Setting safe, powerful defaults for Transolver
            self.learned_model = transolver.Model(
                in_dim=params.get('input_size', 8),  # 8 Features (pos + normals + curvature + is_surface)
                out_dim=params.get('output_size', 1), # 1 Output (Log10 Life)
                n_hidden=params.get('n_hidden', 128), # Embedding dimension
                n_layers=params.get('n_layers', 4),   # Number of Transformer blocks
                n_head=params.get('n_head', 8),       # Attention heads
                slice_num=params.get('slice_num', 32),# Physics slicing
                dropout=params.get('dropout', 0.0),
                act='gelu',
                mlp_ratio=4
            )
        else:
            raise ValueError(f"Unsupported core model: {self.core_model_name}")

    def forward(self, inputs, is_training):
        # Transolver expects [Batch, N, Features], unlike DGCNN which expects [Batch, Features, N]
        processed_features, had_no_batch_dim = self.process_inputs(inputs)
        
        # Pass through Transolver -> Output shape: [Batch, N, out_dim]
        raw_network_output = self.learned_model(processed_features)
        
        # If the input was passed without a batch dimension (e.g., [N, Features]),
        # we must strip the batch dimension from the output to match targets.
        if had_no_batch_dim:
            raw_network_output = raw_network_output.squeeze(0) # -> [N, out_dim]
        
        # 1. Training / Classifier returns raw math
        if is_training or 'classifier' in self.purpose:
            return raw_network_output
            
        # 2. Validation / Evaluation denormalizes to real Log10 Life automatically
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
        Processes inputs, selectively normalizes them, and prepares features.
        Transolver requires shape: [Batch, N, Features].
        """
        device = next(self.parameters()).device
        pos = inputs['mesh_pos'].to(device)

        # Check if batch dimension exists (train.py squeeze_data_frame removes it)
        has_batch = pos.dim() == 3
        
        # 1. Normalize Coordinates (Zero mean, Unit variance)
        if not has_batch: # Shape: (N, 3)
            pos_norm = (pos - pos.mean(dim=0)) / (pos.std(dim=0) + 1e-8)
        else:             # Shape: (Batch, N, 3)
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

        # 5. Add Distance to Step (NORMALIZE IT!)
        if 'dist_to_step' in inputs:
            dist = inputs['dist_to_step'].to(device)
            # Normalize to keep the math stable
            if not has_batch:
                dist_norm = (dist - dist.mean(dim=0)) / (dist.std(dim=0) + 1e-8)
            else:
                dist_norm = (dist - dist.mean(dim=1, keepdim=True)) / (dist.std(dim=1, keepdim=True) + 1e-8)
            features_list.append(dist_norm)

        # 4. Add Surface Mask (DO NOT NORMALIZE)
        if 'is_surface' in inputs:
            features_list.append(inputs['is_surface'].to(device))

        # 5. Concatenate everything -> Shape: (N, Features) or (B, N, Features)
        combined_features = torch.cat(features_list, dim=-1)

        # 6. Ensure (Batch, N, Channels) format for Transolver
        if not has_batch: 
            combined_features = combined_features.unsqueeze(0) # -> (1, N, 8)
            had_no_batch_dim = True
        else:
            had_no_batch_dim = False
            
        return combined_features, had_no_batch_dim
     
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
        # Load weights into the transolver model safely
        self.learned_model.load_state_dict(torch.load(path + "_learned_model.pth", map_location=next(self.parameters()).device))
        if 'regressor' in self.purpose:
            try:
                self._output_life_normalizer = torch.load(path + "_output_life_normalizer.pth")
                print("Loaded model + normalizer.")
            except:
                print("Warning: Normalizer file not found.")
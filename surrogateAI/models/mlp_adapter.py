import torch
from torch import nn as nn
from models import normalization
from models import mlp

class Model(nn.Module):
    def __init__(self, params, core_model_name="mlp"):
        super(Model, self).__init__()

        self._params = params
        self.purpose = params.get("purpose", "regressor")

        if "regressor" in self.purpose:
            self._output_life_normalizer = normalization.Normalizer(
                size=1,
                name="output_life_normalizer"
            )
        else:
            self._output_life_normalizer = None

        self.learned_model = mlp.MLP(
            in_dim=params.get("input_size", 9),
            hidden_dim=params.get("hidden_dim", 256),
            n_layers=params.get("n_layers", 6),
            out_dim=params.get("output_size", 1),
            dropout=params.get("dropout", 0.1),
        )

    def forward(self, inputs, is_training):
        processed_features, had_no_batch_dim = self.process_inputs(inputs)

        raw_network_output = self.learned_model(processed_features)

        if had_no_batch_dim:
            raw_network_output = raw_network_output.squeeze(0)

        if is_training or "classifier" in self.purpose:
            return raw_network_output
        else:
            return self._update(raw_network_output)

    def _update(self, per_node_network_output):
        fatigue_pred = self._output_life_normalizer.inverse(
            per_node_network_output[:, :]
        )
        return fatigue_pred

    def process_inputs(self, inputs):
        device = next(self.parameters()).device
        pos = inputs["mesh_pos"].to(device)
        has_batch = pos.dim() == 3

        # 1. Normalize Coordinates
        if not has_batch:
            pos_norm = (pos - pos.mean(0)) / (pos.std(0) + 1e-8)
        else:
            pos_norm = (pos - pos.mean(1, keepdim=True)) / (pos.std(1, keepdim=True) + 1e-8)

        features_list = [pos_norm]

        # 2. Add Normals (Raw)
        if "normals" in inputs:
            features_list.append(inputs["normals"].to(device))

        # 3. Add Curvature (Normalized)
        if "curvature" in inputs:
            curv = inputs["curvature"].to(device)
            if not has_batch:
                curv_norm = (curv - curv.mean(0)) / (curv.std(0) + 1e-8)
            else:
                curv_norm = (curv - curv.mean(1, keepdim=True)) / (curv.std(1, keepdim=True) + 1e-8)
            features_list.append(curv_norm)

        # 4. Add Distance to Step (Normalized)
        if "dist_to_step" in inputs:
            dist = inputs["dist_to_step"].to(device)
            if not has_batch:
                dist_norm = (dist - dist.mean(0)) / (dist.std(0) + 1e-8)
            else:
                dist_norm = (dist - dist.mean(1, keepdim=True)) / (dist.std(1, keepdim=True) + 1e-8)
            features_list.append(dist_norm)

        # 5. Add Surface Mask (Raw)
        if "is_surface" in inputs:
            features_list.append(inputs["is_surface"].to(device))

        # Combine all features
        combined_features = torch.cat(features_list, dim=-1)

        if combined_features.dim() == 2:
            x = combined_features.unsqueeze(0)
            had_no_batch_dim = True
        else:
            x = combined_features
            had_no_batch_dim = False

        return x, had_no_batch_dim

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

    # --- CRITICAL MISSING METHODS ADDED HERE ---
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
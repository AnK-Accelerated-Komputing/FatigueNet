import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Per-node MLP baseline
    Input  : [B, N, F]
    Output : [B, N, 1]
    """
    def __init__(self, in_dim=9, hidden_dim=256, n_layers=6, out_dim=1, dropout=0.1):
        super().__init__()

        layers = []
        dim = in_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, N, F]
        B, N, F = x.shape
        
        # Flatten batch and nodes together to process as independent points
        x = x.reshape(B * N, F)
        
        out = self.net(x)
        
        # Reshape back to original dimensions
        out = out.reshape(B, N, -1)
        return out
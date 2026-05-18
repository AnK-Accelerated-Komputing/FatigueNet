

'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""
import torch
import torch.nn as nn
from typing import Tuple

device = torch.device('cuda')

def intermediate(x, xx):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    torch.cuda.empty_cache()
    return -xx - inner

def knn(x, k):
    x = x.to(torch.float16)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = intermediate(x, xx) - xx.transpose(2, 1)
    torch.cuda.empty_cache()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    torch.cuda.empty_cache()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class regDGCNN_seg(nn.Module):
    """
    Graph-based segmentation network using Dynamic Graph CNN (DGCNN) blocks.
    Processes point cloud inputs and produces per-point predictions.

    Args:
        output_size (int): Number of output channels per point (e.g., classes or regression dims).
        input_dims (int): Dimensionality of input point features.
        k (int): Number of nearest neighbors for graph construction.
        emb_dims (int): Embedding dimensionality in the bottleneck layer.
        dropout (float): Dropout probability for final classifier.
    """

    def __init__(
        self,
        output_size: int = 3,
        input_dims: int = 6,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.output_size = output_size
        self.input_dims = input_dims
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

        # GroupNorm layers for 2D convolutions (graph feature extractors)
        self.bn1a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn1b  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn2a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn2b  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn3a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn3b = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn4a = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn4b = nn.GroupNorm(num_groups=8,  num_channels=64)

        # GroupNorm layers for 1D convolutions 
        self.bn5   = nn.GroupNorm(num_groups=16, num_channels=emb_dims)  
        self.bn6   = nn.GroupNorm(num_groups=32, num_channels=1024)
        self.bn7  = nn.GroupNorm(num_groups=16, num_channels=512)
        self.bn8   = nn.GroupNorm(num_groups=16, num_channels=256)
        
        # Graph feature extraction layers
        self.conv1a = nn.Sequential(
            nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False),
            self.bn1a,
            nn.LeakyReLU(0.2)
        )  # First-edge features
        self.conv1b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn1b,
            nn.LeakyReLU(0.2)
        )  # Second-edge features

        # Second graph block uses concatenated features
        self.conv2a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2a,
            nn.LeakyReLU(0.2)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2b,
            nn.LeakyReLU(0.2)
        )

        # Third graph block
        self.conv3a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3a,
            nn.LeakyReLU(0.2)
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn3b,
            nn.LeakyReLU(0.2)
        )

        # Fourth graph block with two parallel convs
        self.conv4a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn4a,
            nn.LeakyReLU(0.2)
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4b,
            nn.LeakyReLU(0.2)
        )

        # Embedding layer - aggregates concatenated graph outputs
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 * 4, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(0.2)
        )

        # Final per-point classifier
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims + 64 * 4, 1024, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(0.2)
        )
        self.dp1 = nn.Dropout(self.dropout)
        self.conv9 = nn.Conv1d(256, output_size, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            inputs: Raw point cloud tensor (e.g., [B, dims, N]).
            device: Torch device for processing.

        Returns:
            Per-point predictions of shape [B, output_size, N] or [N, output_size] if B==1.
        """
        # print('reached here')
        # Preprocess inputs (e.g., feature selection or encoding)
        B, _, N = inputs.size()

        # ---- Graph block 1 ----
        x1 = self._dgcnn_block(inputs, k=self.k, convs=[self.conv1a, self.conv1b])
        # ---- Graph block 2 ----
        x2 = self._dgcnn_block(x1, k=self.k, convs=[self.conv2a, self.conv2b])
        # ---- Graph block 3 ----
        x3 = self._dgcnn_block(x2, k=self.k, convs=[self.conv3a, self.conv3b])
        # ---- Graph block 4 (parallel paths) ----
        x4 = self._dgcnn_block(x3, k=self.k, convs=[self.conv4a, self.conv4b])

        # Concatenate all local features
        local_feats = torch.cat([x1, x2, x3, x4], dim=1)  # [

        # Global embedding
        emb = self.conv5(local_feats)   # [B, emb_dims, N]
        emb = torch.max(emb, -1, keepdim=True)[0]  # [B, emb_dims, 1]
        emb = emb.repeat(1, 1, N)  # Broadcast to all points

        # Combine global and local features for classification
        feats = torch.cat([emb, local_feats], dim=1)  # [B, emb_dims+64*4, N]

        # Classification head
        x = self.conv6(feats)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x) 

        # If batch size is 1, return [N, output_size]
        if B == 1:
            x = x.squeeze(0).permute(1, 0)
        return x

    def _dgcnn_block(
        self,
        x: torch.Tensor,
        k: int,
        convs: Tuple[nn.Sequential, nn.Sequential]
    ) -> torch.Tensor:
        """
        Helper for two-layer DGCNN block: feature extraction, neighbor aggregation, and max pooling.
        """
        # Build graph features: [B, C*2, N, k]
        x = get_graph_feature(x, k=k)
        # Apply first conv + activation
        x = convs[0](x)
        # Apply second conv + activation
        x = convs[1](x)
        # Aggregate via max over neighbors
        return x.max(dim=-1, keepdim=False)[0]

    def _single_dgcnn(
        self,
        x: torch.Tensor,
        k: int,
        conv: nn.Sequential
    ) -> torch.Tensor:
        """
        Single-layer DGCNN block: conv + max pooling over k neighbors.
        """
        x = get_graph_feature(x, k=k)
        x = conv(x)
        return x.max(dim=-1, keepdim=False)[0]


'''






#......................................Physics and k - neighbour features separately..............................................
#....................................................................................................................................
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

This module is used to define both point-cloud based and graph-based models, including RegDGCNN, PointNet, and several Graph Neural Network (GNN) models
for the task of surrogate modeling of the aerodynamic drag.
"""
import torch
import torch.nn as nn
from typing import Tuple

device = torch.device('cuda')

def intermediate(x, xx):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    torch.cuda.empty_cache()
    return -xx - inner

def knn(x, k):
    x = x.to(torch.float16)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = intermediate(x, xx) - xx.transpose(2, 1)
    torch.cuda.empty_cache()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    torch.cuda.empty_cache()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class regDGCNN_seg(nn.Module):
    """
    Graph-based segmentation network using Dynamic Graph CNN (DGCNN) blocks.
    Processes point cloud inputs and produces per-point predictions.

    Args:
        output_size (int): Number of output channels per point (e.g., classes or regression dims).
        input_dims (int): Dimensionality of input point features.
        k (int): Number of nearest neighbors for graph construction.
        emb_dims (int): Embedding dimensionality in the bottleneck layer.
        dropout (float): Dropout probability for final classifier.
    """

    def __init__(
        self,
        output_size: int = 3,
        input_dims: int = 6,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.output_size = output_size
        self.input_dims = input_dims
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

        # GroupNorm layers for 2D convolutions (graph feature extractors)
        self.bn1a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn1b  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn2a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn2b  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn3a  = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn3b = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn4a = nn.GroupNorm(num_groups=8,  num_channels=64)
        self.bn4b = nn.GroupNorm(num_groups=8,  num_channels=64)

        # GroupNorm layers for 1D convolutions 
        self.bn5   = nn.GroupNorm(num_groups=16, num_channels=emb_dims)  
        self.bn6   = nn.GroupNorm(num_groups=32, num_channels=1024)
        self.bn7  = nn.GroupNorm(num_groups=16, num_channels=512)
        self.bn8   = nn.GroupNorm(num_groups=16, num_channels=256)
        
        # Graph feature extraction layers
        self.conv1a = nn.Sequential(
            nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False),
            self.bn1a,
            nn.LeakyReLU(0.2)
        )  # First-edge features
        self.conv1b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn1b,
            nn.LeakyReLU(0.2)
        )  # Second-edge features

        # Second graph block uses concatenated features
        self.conv2a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2a,
            nn.LeakyReLU(0.2)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2b,
            nn.LeakyReLU(0.2)
        )

        # Third graph block
        self.conv3a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3a,
            nn.LeakyReLU(0.2)
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn3b,
            nn.LeakyReLU(0.2)
        )

        # Fourth graph block with two parallel convs
        self.conv4a = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn4a,
            nn.LeakyReLU(0.2)
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4b,
            nn.LeakyReLU(0.2)
        )

        # Embedding layer - aggregates concatenated graph outputs
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 * 4, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(0.2)
        )

        # Final per-point classifier
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims + 64 * 4, 1024, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(0.2)
        )
        self.dp1 = nn.Dropout(self.dropout)
        self.conv9 = nn.Conv1d(256, output_size, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.
            Args:
                inputs: Raw point cloud tensor (e.g., [B, dims, N]).
                device: Torch device for processing.

            Returns:
                Per-point predictions of shape [B, output_size, N] or [N, output_size] if B==1.
            """
            B, _, N = inputs.size()

            # --- STRATEGY 3: THE PHYSICS FIX ---
            # 1. Extract ONLY the spatial coordinates (X, Y, Z are the first 3 channels)
            spatial_coords = inputs[:, :3, :]
            
            # 2. Calculate the physical neighborhood graph ONLY on XYZ.
            # This prevents nodes from teleporting based on curvature or normal vectors.
            physical_idx = knn(spatial_coords, k=self.k)
            # -----------------------------------

            # ---- Graph block 1 ----
            # Pass the physical_idx to force the network to use the physical mesh
            x1 = self._dgcnn_block(inputs, k=self.k, convs=[self.conv1a, self.conv1b], idx=physical_idx)
            # ---- Graph block 2 ----
            x2 = self._dgcnn_block(x1, k=self.k, convs=[self.conv2a, self.conv2b], idx=physical_idx)
            # ---- Graph block 3 ----
            x3 = self._dgcnn_block(x2, k=self.k, convs=[self.conv3a, self.conv3b], idx=physical_idx)
            # ---- Graph block 4 (parallel paths) ----
            x4 = self._dgcnn_block(x3, k=self.k, convs=[self.conv4a, self.conv4b], idx=physical_idx)

            # Concatenate all local features
            local_feats = torch.cat([x1, x2, x3, x4], dim=1)  

            # Global embedding
            emb = self.conv5(local_feats)   # [B, emb_dims, N]
            emb = torch.max(emb, -1, keepdim=True)[0]  # [B, emb_dims, 1]
            emb = emb.repeat(1, 1, N)  # Broadcast to all points

            # Combine global and local features for classification
            feats = torch.cat([emb, local_feats], dim=1)  # [B, emb_dims+64*4, N]

            # Classification head
            x = self.conv6(feats)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.dp1(x)
            x = self.conv9(x) 

            # If batch size is 1, return [N, output_size]
            if B == 1:
                x = x.squeeze(0).permute(1, 0)
            return x

    def _dgcnn_block(
            self,
            x: torch.Tensor,
            k: int,
            convs: Tuple[nn.Sequential, nn.Sequential],
            idx: torch.Tensor = None  # <-- Added idx argument here
        ) -> torch.Tensor:
            """
            Helper for two-layer DGCNN block: feature extraction, neighbor aggregation, and max pooling.
            """
            # Build graph features, explicitly using the passed physical graph (idx)
            x = get_graph_feature(x, k=k, idx=idx)
            # Apply first conv + activation
            x = convs[0](x)
            # Apply second conv + activation
            x = convs[1](x)
            # Aggregate via max over neighbors
            return x.max(dim=-1, keepdim=False)[0]

    def _single_dgcnn(
        self,
        x: torch.Tensor,
        k: int,
        conv: nn.Sequential
    ) -> torch.Tensor:
        """
        Single-layer DGCNN block: conv + max pooling over k neighbors.
        """
        x = get_graph_feature(x, k=k)
        x = conv(x)
        return x.max(dim=-1, keepdim=False)[0]






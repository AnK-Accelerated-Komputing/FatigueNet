# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Online data normalization."""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, size, name, max_accumulations=10**6, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self._name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False).to(device)

        self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(device)
        self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(device)
        self._acc_sum = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(device)
        self._acc_sum_squared = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    # --- THIS IS THE NEW, REQUIRED METHOD ---
    def set_stats(self, mean, std):
        """
        Sets the statistics of the normalizer directly from pre-calculated
        mean and standard deviation, bypassing the online accumulation.
        
        Args:
            mean (torch.Tensor): The pre-calculated mean of the data.
            std (torch.Tensor): The pre-calculated standard deviation of the data.
        """
        # Set a dummy count of 1. This is a stable way to inject the stats.
        self._acc_count.fill_(1.0)
        
        # Set the sum to be the mean (since mean = sum / count)
        self._acc_sum = mean.clone().detach().to(device)
        
        # Calculate the sum of squares from the mean and std.
        # variance = E[X^2] - (E[X])^2  -->  E[X^2] = variance + (E[X])^2
        # Since count is 1, sum_squared is E[X^2].
        variance = std**2
        self._acc_sum_squared = (variance + mean**2).clone().detach().to(device)
        
        # Set accumulations to max to prevent further updates
        self._num_accumulations.fill_(self._max_accumulations)
        print(f"Statistics for '{self._name}' set directly. Mean={mean.item():.4f}, Std={std.item():.4f}")
    # --- END OF NEW METHOD ---

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(batched_data.shape[0], dtype=torch.float32, device=device)
        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data**2, dim=0)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count)
        self._num_accumulations = self._num_accumulations.add(1.)

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=device))
        # variance = E[X^2] - (E[X])^2
        diff = (self._acc_sum_squared / safe_count) - self._mean()**2
        # Clamp to avoid negative values due to floating point inaccuracies
        diff = torch.clamp(diff, min=0.0)
        std = torch.sqrt(diff)
        return torch.maximum(std, self._std_epsilon)

    def get_acc_sum(self):
        return self._acc_sum

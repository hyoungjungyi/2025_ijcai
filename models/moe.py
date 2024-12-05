import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class MOEModel(nn.Module):
    def __init__(self, input_size, experts, train_experts=False):
        """
        Mixture of Experts (MOE) Model

        Args:
            input_size (int): Input feature size.
            experts (list of nn.Module): List of pre-trained expert models.
            train_experts (bool): Whether to train expert models or freeze them.
        """
        super(MOEModel, self).__init__()
        self.num_experts = len(experts)
        self.gating_network = GatingNetwork(input_size, self.num_experts)  # Gating network
        self.experts = nn.ModuleList(experts)  # Pre-trained experts
        if not train_experts:
            # Freeze experts' parameters
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        Forward pass for MOE.

        Args:
            batch_x (torch.Tensor): Input sequence of shape [batch_size, seq_len, feature].
            batch_x_mark (torch.Tensor): Time encoding for the input sequence of shape [batch_size, seq_len, time_feature].
            dec_inp (torch.Tensor): Decoder input of shape [batch_size, pred_len, feature].
            batch_y_mark (torch.Tensor): Time encoding for the target sequence of shape [batch_size, pred_len, time_feature].

        Returns:
            output (torch.Tensor): Weighted output of experts, shape [batch_size, seq_len, 1].
            gating_weights (torch.Tensor): Gating network's weights for each expert, shape [batch_size, num_experts].
        """
        # Average pooling across the sequence dimension for gating input
        gating_input = torch.mean(batch_x, dim=1)

        # Get gating weights for  each `tic`
        gating_weights_tic = self.gating_network(gating_input)  # (batch_size, num_experts)

        # Aggregate gating weights across `tic` to get date-level weights
        gating_weights = gating_weights_tic.mean(dim=0, keepdim=True)


        # Get outputs from each expert
        expert_outputs = torch.cat(
            [expert(batch_x, batch_x_mark, dec_inp, batch_y_mark) for expert in self.experts],
            dim=1
        )  # (batch_size, num_experts,1)

        # Weighted sum of expert outputs
        output = torch.sum(gating_weights * expert_outputs, dim=1)
        return output, gating_weights


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        """
        Gating network for MOE.

        Args:
            input_size (int): Input feature size.
            num_experts (int): Number of experts (e.g., daily, weekly, monthly).
        """
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)  # Output weights for each expert
        )

    def forward(self, x):
        """
        Forward pass for the gating network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature).

        Returns:
            torch.Tensor: Gating weights, shape (batch_size, num_experts).
        """
        return self.network(x)


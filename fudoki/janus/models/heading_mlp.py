import torch
import torch.nn as nn


class TrajectoryHeadingMLP(nn.Module):
    """
    input: [B, 8, 2]
    output: [B, 8, 1]
    """
    def __init__(self, hidden_dims=[512, 512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []
        input_dim = 8 * 2

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))   
            layers.append(nn.GELU())             
            layers.append(nn.Dropout(dropout))   
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, traj):
        x = traj.reshape(traj.size(0), -1)      # [B, 8, 2] → [B, 16]
        out = self.net(x)
        out = torch.tanh(out) * 3.14159
        return out.unsqueeze(-1)                # [B, 8, 1]
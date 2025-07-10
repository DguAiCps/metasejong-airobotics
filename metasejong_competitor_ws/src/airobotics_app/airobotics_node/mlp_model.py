# === mlp_model.py ===
import torch.nn as nn

class ResidualMLP(nn.Module):
    """잔차 오차 보정용 MLP"""
    def __init__(self, input_size=4, output_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)


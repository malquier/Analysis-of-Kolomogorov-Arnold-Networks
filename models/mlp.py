import torch.nn as nn


# -------------------------
# Simple MLP baseline
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=3, out_dim=1, act="tanh"):
        super().__init__()
        layers = []
        d = in_dim
        activation = nn.Tanh() if act == "tanh" else nn.ReLU()
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), activation]
            d = hidden
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

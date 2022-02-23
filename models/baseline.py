import torch
from torch.nn import Linear
from torch_geometric.nn import global_add_pool

class Baseline(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = Linear(in_channels, out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, data):
        x = data.x
        x = global_add_pool(x, data.batch)
        x = self.lin(x)
        x = self.act(x)
        return x


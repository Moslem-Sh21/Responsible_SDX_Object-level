import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=2048, output_ch=2048, resolution=1, nonlinearity="relu"):
        super(MLP, self).__init__()
        # SDXL uses 2048 hidden dimensions compared to 1280 in SD v1.4
        output_dim = output_ch * resolution * resolution
        self.resolution = resolution
        self.output_ch = output_ch
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)


class MLPXL(nn.Module):
    """
    Enhanced MLP for SDXL with deeper architecture and better feature processing
    """
    def __init__(self, input_dim=100, hidden_dim=2048, output_ch=2048, resolution=1, nonlinearity="silu"):
        super(MLPXL, self).__init__()
        # SDXL uses larger hidden dimensions
        output_dim = output_ch * resolution * resolution
        self.resolution = resolution
        self.output_ch = output_ch
        
        # Multi-layer architecture for better feature processing
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Activation function (SDXL typically uses SiLU)
        if nonlinearity == "silu":
            self.activation = nn.SiLU()
        elif nonlinearity == "relu":
            self.activation = nn.ReLU()
        elif nonlinearity == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()  # Default
            
        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        
        # Multi-layer processing
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)


model_types = {
    "MLP": MLP,
    "MLPXL": MLPXL,  # Enhanced version for SDXL
}
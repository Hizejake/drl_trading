import torch
import torch.nn as nn

class CVML(nn.Module):
    """
    Convolutional Cross-Variate Mixing Layer (CVML) for Limit Order Book data.
    Takes flat (40,) features representing 10 levels of (bid_p, ask_p, bid_s, ask_s)
    and reconstructs them into 2D spatial features for depthwise and pointwise convolution.
    """
    def __init__(self, in_channels=4, levels=10, out_dim=64):
        super(CVML, self).__init__()
        self.levels = levels
        self.in_channels = in_channels
        
        # Depthwise convolution along the price levels (groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(3, 1), # convolve along 3 adjacent levels
            padding=(1, 0),
            groups=self.in_channels
        )
        
        # Pointwise convolution to mix the 4 variates (bid price, bid size, ask price, ask size)
        self.pointwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=16,
            kernel_size=1
        )
        
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Output dimension: 16 channels * 10 levels * 1 width = 160 flat features
        # Project down to out_dim (64)
        self.projection = nn.Linear(16 * self.levels, out_dim)
        
    def forward(self, x):
        # x input shape is (Batch, 40)
        batch_size = x.shape[0]
        
        # Note: Depending on the specific dataset ordering, you must reshape carefully.
        # Assuming the 40 features are: 
        # [bid_p1..p10, ask_p1..p10, bid_s1..s10, ask_s1..s10]
        # Reshape to (Batch, Channels=4, Levels=10, Width=1)
        x_reshaped = x.view(batch_size, self.in_channels, self.levels, 1)
        
        # Apply depthwise
        out = self.depthwise(x_reshaped)
        out = self.activation(out)
        
        # Apply pointwise
        out = self.pointwise(out)
        out = self.activation(out)
        
        # Flatten and project
        out = self.flatten(out)
        out = self.projection(out)
        
        return out

if __name__ == "__main__":
    # Quick test of tensor shapes
    model = CVML()
    batch_of_lob_states = torch.randn(32, 40)
    output = model(batch_of_lob_states)
    print(f"Input shape: {batch_of_lob_states.shape}")
    print(f"CVML Output shape: {output.shape}")  # Expected: [32, 64]

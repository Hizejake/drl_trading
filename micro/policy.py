import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from .cvml import CVML

class HierarchicalFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Stable Baselines 3 PPO.
    Takes a Dict observation space with:
    - lob: (40,) flat features
    - macro: (macro_dim,) consensus scalars + pooled semantic embedding
    Passes LOB through CVML, then concatenates with Macro.
    """
    def __init__(self, observation_space: spaces.Dict, cvml_out_dim: int = 64, cnn_activation=nn.ReLU):
        macro_dim = observation_space["macro"].shape[0]
        super(HierarchicalFeatureExtractor, self).__init__(observation_space, features_dim=cvml_out_dim + macro_dim)

        self.cvml = CVML(in_channels=4, levels=10, out_dim=cvml_out_dim)
        
        # Optional: Projection layer to avoid macro vector dominating the CVML features
        # For now, we will just pass the macro vector directly into the concatenation
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Stable Baselines 3 passes the Dict observation as a dictionary of tensors
        lob_data = observations["lob"]
        macro_data = observations["macro"]
        
        # Ensure correct dtype
        lob_data = lob_data.float()
        macro_data = macro_data.float()
        
        # 1. Process LOB through the Convolutional Cross-Variate Mixing Layer
        cvml_features = self.cvml(lob_data)
        
        # 2. Concatenate the temporal microstructure features with the semantic macro features
        # Shape: (Batch, cvml_out_dim + macro_dim)
        combined_features = torch.cat([cvml_features, macro_data], dim=1)
        
        return combined_features

# For pure LOB testing (ablation study)
class FlatMLPFeatureExtractor(BaseFeaturesExtractor):
    """Ablation baseline feature extractor using flat MLPs instead of CVML."""
    def __init__(self, observation_space: spaces.Dict, mlp_out_dim: int = 64):
        macro_dim = observation_space["macro"].shape[0]
        super(FlatMLPFeatureExtractor, self).__init__(observation_space, features_dim=mlp_out_dim + macro_dim)
        self.mlp = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        lob_data = observations["lob"].float()
        macro_data = observations["macro"].float()
        mlp_features = self.mlp(lob_data)
        return torch.cat([mlp_features, macro_data], dim=1)

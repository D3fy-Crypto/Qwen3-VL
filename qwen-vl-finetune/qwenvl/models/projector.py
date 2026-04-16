import torch
import torch.nn as nn


class ProjectorMLP(nn.Module):
    """
    MLP Projector: GRU hidden dim → intermediate → Qwen embedding dim.
    
    Maps trajectory GRU hidden states (256) to Qwen token embeddings (4096).
    Supports variable K to produce K*4096 dimensional outputs per sequence.
    
    Architecture:
        Input (256) → Linear(1024) → GELU → Linear(K*4096) → Output
    """

    def __init__(self, gru_hidden_dim: int = 256, qwen_hidden_dim: int = 4096,
                 intermediate_dim: int = 1024, k: int = 1):
        """
        Args:
            gru_hidden_dim: GRU hidden dimension (default: 256)
            qwen_hidden_dim: Qwen embedding dimension (default: 4096)
            intermediate_dim: Hidden dimension of MLP (default: 1024)
            k: Multiplier for output dimension (output_dim = k * qwen_hidden_dim)
        """
        super().__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        self.intermediate_dim = intermediate_dim
        self.k = k
        self.output_dim = k * qwen_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(gru_hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, gru_hidden_dim) or
               (batch_size, seq_len, gru_hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim) or
            (batch_size, seq_len, output_dim)
        """
        return self.net(x)

    def get_output_shape(self, batch_size: int, seq_len: int = None) -> tuple:
        """Helper to compute output shape."""
        if seq_len is not None:
            return (batch_size, seq_len, self.output_dim)
        return (batch_size, self.output_dim)

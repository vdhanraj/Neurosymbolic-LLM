from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dtype = dtype
        self.encoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        self.layer_id = layer_id

    def forward(self, x):
        out = self.encoder_layer(x)
        return out
class Decoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dtype = dtype
        self.decoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        self.layer_id = layer_id

    def forward(self, x):
        out = self.decoder_layer(x)
        return out

class Encoder_Deep(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, hidden_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dtype = dtype

        self.encoder_layer_1 = nn.Linear(input_dim,  hidden_dim, bias=bias, dtype=dtype)
        self.encoder_layer_2 = nn.Linear(hidden_dim, output_dim, bias=bias, dtype=dtype)
        self.layer_id = layer_id

    def forward(self, x):
        x   = self.encoder_layer_1(x)
        out = self.encoder_layer_2(x)
        return out
class Decoder_Deep(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, hidden_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dtype = dtype

        self.decoder_layer_1 = nn.Linear(input_dim,  hidden_dim, bias=bias, dtype=dtype)
        self.decoder_layer_2 = nn.Linear(hidden_dim, output_dim, bias=bias, dtype=dtype)
        self.layer_id = layer_id

    def forward(self, x):
        x   = self.decoder_layer_1(x)
        out = self.decoder_layer_2(x)
        return out

    
class LastTokenTransformer(nn.Module):
    def __init__(self, layer_id, data_dim, output_dim, num_layers=4,
                 num_heads=8, hidden_dim=512, dropout=0.1, dtype=torch.bfloat16):
        """
        Transformer model that processes LLM hidden states and outputs a semantic vector.

        Args:
            data_dim (int): Dimension of the input hidden states.
            output_dim (int): Dimension of the final semantic output.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Size of the feedforward hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.layer_id = layer_id
        self.data_dim = data_dim
        self.output_dim = output_dim

        # Input projection layer (embedding input into model hidden size)
        self.input_proj = nn.Linear(data_dim, hidden_dim, dtype=dtype)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            dtype=dtype
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, sequence_number, data_dim)

        Returns:
            Tensor: Output tensor of shape (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_proj(x)  # Shape: (batch, seq_len, hidden_dim)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch, seq_len, hidden_dim)

        # Select the last token's hidden state
        last_token = x[:, -1, :]  # Shape: (batch, hidden_dim)

        # Output projection
        out = self.output_proj(last_token)  # Shape: (batch, output_dim)

        return out

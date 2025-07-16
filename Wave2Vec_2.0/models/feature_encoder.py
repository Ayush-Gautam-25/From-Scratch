import torch 
from torch import nn
from typing import Optional

class CNNFeatureEncoder(nn.Module):
    def __init__(self, input_dim: Optional[int] = 1, conv_dim: Optional[int] = 512):
        super().__init__()

        def conv_block(in_dim, out_dim, k, s, p):
            return nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
                nn.GELU()
            )
        # Taken from wave2vec 2.0 paper
        self.conv_layers = nn.Sequential(
            conv_block(input_dim, conv_dim, 10, 5, 3),
            conv_block(conv_dim, conv_dim, 3, 2, 1),
            # conv_block(conv_dim, conv_dim, 3, 2, 1),
            # conv_block(conv_dim, conv_dim, 3, 2, 1),
            # conv_block(conv_dim, conv_dim, 3, 2, 1),
            # conv_block(conv_dim, conv_dim, 2, 2, 0),
            # conv_block(conv_dim, conv_dim, 2, 2, 0),
        )

        self.layer_norm = nn.LayerNorm(conv_dim)


    def forward(self, x):
        """
        Args:
            x: waveform tensor [B, T] or [B, 1, T]

        Returns:
            z: latent feature map [B, T', D]
        """

        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = x.transpose(1, 2) 
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    model = CNNFeatureEncoder()
    dummy_wave = torch.randn(4, 16000)  # 4 audio clips of 1 second (16kHz)
    z = model(dummy_wave)
    print(z.shape)

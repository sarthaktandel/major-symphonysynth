import torch
import torch.nn as nn


class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        residual = x

        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)

        gate = torch.sigmoid(self.gate(x))

        x = gate * x_conv + (1 - gate) * x
        x = self.norm(x + residual)

        return x


class MelodyTransformerMamba(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 👇 Mamba-like block
        self.mamba = SimpleMambaBlock(d_model)

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer(x)
        x = self.mamba(x)
        return self.output(x)
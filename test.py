import torch
from src.models.melody_model_mamba import MelodyTransformerMamba

model = MelodyTransformerMamba(vocab_size=400)
x = torch.randint(0, 400, (1, 128))

out = model(x)

print(out.shape)
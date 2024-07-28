import torch.nn as nn
import torch
from layers.Embed import DataEmbedding
from layers.Kan import KANLinear


class Kan(nn.Module):
    def __init__(self, c_in, d_model, pred_len):
        super().__init__()

        self.embed = DataEmbedding(c_in, d_model)
        self.kan = nn.Sequential(
            KANLinear(d_model, d_model * 4),
            nn.ReLU(),
            KANLinear(d_model * 4, d_model),
            nn.ReLU(),
            KANLinear(d_model, pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        x = self.embed(x_enc, x_mark_enc)
        x = self.kan(x)
        x = x[:, -1, :]
        return x


class Linear(nn.Module):
    def __init__(self, c_in, d_model, pred_len):
        super().__init__()
        self.embed = DataEmbedding(c_in, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        x = self.embed(x_enc, x_mark_enc)
        x = self.fc(x)
        x = x[:, -1, :]
        return x


if __name__ == '__main__':
    x_enc = torch.randn((16, 10, 6))
    x_mark_enc = torch.ones((16, 10, 4))
    model = Linear(6, 64, 2)
    out = model(x_enc, x_mark_enc)
    print(out.shape)

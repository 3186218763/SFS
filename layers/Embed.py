import torch.nn as nn
import torch


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, weekday_size=7, day_size=32, month_size=13, year_size=10):
        super().__init__()

        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)
        self.year_embed = nn.Embedding(year_size, d_model)

    def forward(self, x):
        x = x.long()

        # 'year', 'month', 'day', 'weekday'
        year_x = self.year_embed(x[:, :, 0])
        month_x = self.month_embed(x[:, :, 1])
        day_x = self.day_embed(x[:, :, 2])
        weekday_x = self.weekday_embed(x[:, :, 3])

        return year_x + month_x + day_x + weekday_x


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        x_mark = self.temporal_embedding(x_mark)

        out = self.dropout(x + x_mark)

        return out

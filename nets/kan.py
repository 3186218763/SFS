import torch.nn as nn
import torch
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding
from layers.Kan import KANLinear
from layers.AutoCorrelation import MultiHeadCompression


class Kan_Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            KANLinear(d_model, d_model * 4),
            nn.GELU(),
            KANLinear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = x
        x = self.net(x)
        return x + res


class Kan(nn.Module):
    def __init__(self, c_in, d_model, pred_len):
        super().__init__()

        self.embed = DataEmbedding(c_in, d_model)
        self.kan = nn.Sequential(
            Kan_Block(d_model),
            Kan_Block(d_model),
            Kan_Block(d_model),
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


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, d_model, individual=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = 4

        kernel_size = 25
        self.embed = DataEmbedding_wo_pos(c_in=enc_in, d_model=d_model)
        self.decomposition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([KANLinear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList([KANLinear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear_Seasonal = KANLinear(self.seq_len, self.pred_len)
            self.Linear_Trend = KANLinear(self.seq_len, self.pred_len)
        self.multiHeadCompression = MultiHeadCompression(d_model)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.embed(x, x_mark_enc)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        x = self.multiHeadCompression(x)
        return x


if __name__ == '__main__':
    x_enc = torch.randn((16, 10, 6))
    x_mark_enc = torch.ones((16, 10, 4))
    model = Linear(6, 64, 2)
    out = model(x_enc, x_mark_enc)
    print(out.shape)

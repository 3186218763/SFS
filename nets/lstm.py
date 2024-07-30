import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class LSTM(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, pred_len, num_layers, dropout):
        super().__init__()
        self.data_embedding = DataEmbedding(input_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.data_embedding(x_enc, x_mark_enc)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, pred_len, num_layers, dropout):
        super().__init__()
        self.data_embedding = nn.Linear(input_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),  # *2 because of bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.data_embedding(x_enc)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last timestep's output
        out = self.fc(x)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, pred_len, num_layers, dropout):
        super().__init__()
        self.data_embedding = nn.Linear(input_size, d_model)
        self.gru = nn.GRU(d_model, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.data_embedding(x_enc)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out


class BiGRU(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, pred_len, num_layers, dropout):
        super().__init__()
        self.data_embedding = nn.Linear(input_size, d_model)
        self.gru = nn.GRU(d_model, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),  # *2 because of bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.data_embedding(x_enc)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take the last timestep's output
        out = self.fc(x)
        return out


if __name__ == '__main__':
    input_size = 7
    d_model = 128
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    seq_len = 230
    pred_len = 2
    tensor = torch.randn(1, seq_len, input_size, dtype=torch.float32)
    model = BiGRU(input_size, d_model, hidden_size, pred_len, num_layers, dropout)
    out = model(tensor)
    print(out.shape)

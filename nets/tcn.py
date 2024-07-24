import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_Module(nn.Module):
    def __init__(self,
                 input_size,
                 num_channels,
                 kernel_size,
                 dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 pred_len,
                 kernel_size,
                 dropout):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True, bidirectional=True)
        self.tcn = TCN_Module(lstm_hidden_size * 2, [lstm_hidden_size * 2, lstm_hidden_size * 2, lstm_hidden_size * 2], kernel_size, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # 修改为 lstm_hidden_size * 2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 4, pred_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x, _ = self.bilstm(x_enc)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = x[:, -1, :]  # 提取最后一个时间步的输出

        output = self.mlp(output)
        return output  # 返回最终预测结果



if __name__ == '__main__':
    seq_len = 230
    input_size = 7
    lstm_hidden_size = 515
    lstm_num_layers = 3
    pred_len = 10
    kernel_size = 2
    dropout = 0.1
    tensor = torch.randn(1, seq_len, input_size, dtype=torch.float32)
    model = TCN_LSTM(input_size, lstm_hidden_size, lstm_num_layers, pred_len, kernel_size, dropout)
    out = model(tensor)
    print(out.shape)

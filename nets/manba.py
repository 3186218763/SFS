import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

batch_size = 16


class S6(nn.Module):
    def __init__(self,
                 seq_len,
                 input_size,
                 state_size,
                 device,
                 DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM=True):
        super().__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.state_size = state_size
        self.device = device
        self.use = DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM
        # 线性变换
        self.fc1 = nn.Linear(input_size, input_size, device=device)
        self.fc2 = nn.Linear(input_size, state_size, device=device)
        self.fc3 = nn.Linear(input_size, state_size, device=device)

        # 设定超参数
        self.seq_len = seq_len
        self.input_size = input_size
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(input_size, state_size, device=device), p=2, dim=-1))

        # 参数初始化
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.delta = torch.zeros(batch_size, self.seq_len, self.input_size, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.input_size, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.input_size, self.state_size, device=device)

        # 定义内部参数h和y
        self.h = torch.zeros(batch_size, self.seq_len, self.input_size, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.input_size, device=device)

    # 离散化函数
    def discretization(self):
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        # dA = torch.matrix_exp(A * delta)  # matrix_exp() only supports square matrix
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    # 前向传播
    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        # 离散化
        self.discretization()

        if self.use:
            # 如果不使用'h_new'，将触发本地允许错误
            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                # different_batch_size = True
                # 缩放h的维度匹配当前的批次
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB

            else:
                # different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # 改变y的维度
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            # 基于h_new更新h的信息
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:
            # 设置h的维度
            h = torch.zeros(x.size(0), self.seq_len, self.input_size, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # 设置y的维度
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y


class MambaBlock(nn.Module):
    def __init__(self,
                 seq_len,
                 input_size,
                 state_size,
                 device):
        super().__init__()

        self.inp_proj = nn.Linear(input_size, 2 * input_size, device=device)
        self.out_proj = nn.Linear(2 * input_size, input_size, device=device)

        # 残差连接
        self.D = nn.Linear(input_size, 2 * input_size, device=device)

        # 设置偏差属性
        self.out_proj.bias._no_weight_decay = True

        # 初始化偏差
        nn.init.constant_(self.out_proj.bias, 1.0)

        # 初始化S6模块
        self.S6 = S6(seq_len, 2 * input_size, state_size, device)

        # 添加1D卷积
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

        # 添加线性层
        self.conv_linear = nn.Linear(2 * input_size, 2 * input_size, device=device)

        # 正则化
        self.norm = RMSNorm(input_size, device=device)

    # 前向传播
    def forward(self, x):
        x = self.norm(x)

        x_proj = self.inp_proj(x)

        # 1D卷积操作
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)  # Swish激活

        # 线性操作
        x_conv_out = self.conv_linear(x_conv_act)

        # S6模块操作
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish激活

        # 残差连接
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)

        return x_out


class Manba(nn.Module):
    def __init__(self,
                 seq_len,
                 output_size,
                 input_size,
                 num_layers,
                 state_size,
                 device,
                 ):
        super().__init__()

        self.layers = nn.ModuleList(
            [MambaBlock(seq_len, input_size, state_size, device) for _ in range(num_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size // 2, device=device),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size // 2, output_size, device=device),
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, input_size: int, eps: float = 1e-5, device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(input_size, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == '__main__':
    seq_len = 150
    output_size = 2
    input_size = 7
    state_size = 512  # 状态机大小
    num_layers = 4
    device = 'cuda'
    model = Manba(seq_len=seq_len,
                  output_size=output_size,
                  input_size=input_size,
                  num_layers=num_layers,
                  state_size=state_size,
                  device=device)
    tensor = torch.randn(batch_size, seq_len, 7, device=device)
    output = model(tensor)
    print(output.shape)
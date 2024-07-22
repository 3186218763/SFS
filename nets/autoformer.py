import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, Seasonal_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity.
    """

    def __init__(self, seq_len, label_len, pred_len,
                 moving_avg, enc_in, d_model, dropout, n_heads,
                 d_ff, e_layers, d_layers, activation, c_out, factor):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model)
        self.dec_embedding = DataEmbedding_wo_pos(enc_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=Seasonal_Layernorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, c_out, d_ff, moving_avg=moving_avg, dropout=dropout, activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=Seasonal_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len - self.label_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len - self.label_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, :self.label_len, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, :self.label_len, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)

        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out.mean(dim=2)

        return dec_out[:, -self.pred_len:]  # [B, L]


if __name__ == '__main__':
    """
    参数说明
    seq_len: 输入序列的长度。
    label_len: 标签序列的长度（用于趋势初始化）。
    pred_len: 预测序列的长度。
    moving_avg: 移动平均的窗口大小。
    enc_in: 编码器输入的特征维度。
    d_model: 模型的隐藏维度。
    dropout: Dropout概率。
    n_heads: 多头注意力的头数。
    d_ff: 前馈神经网络的维度。
    e_layers: 编码器层数。
    d_layers: 解码器层数。
    activation: 激活函数类型。
    c_out: 输出的特征维度。
    """
    # 测试代码

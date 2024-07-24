import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, pred_len, d_model,
                 n_heads, e_layers, d_layers, d_ff,
                 dropout, factor=5, activation='gelu'):
        super().__init__()
        self.dec_in = dec_in
        self.pred_len = pred_len

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model)
        self.dec_embedding = DataEmbedding(dec_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        x_dec = x_dec.unsqueeze(-1).expand(-1, -1, self.dec_in)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = dec_out.squeeze(-1)

        return dec_out[:, -self.pred_len:]

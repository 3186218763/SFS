import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding


class Reformer(nn.Module):
    """

    """

    def __init__(self, enc_in, d_model, dropout, e_layers,
                 n_heads, bucket_size, n_hashes, d_ff, c_out,
                 label_len, seq_len, pred_len, activation='gelu', ):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.label_len = label_len

        self.fc = nn.Linear(pred_len, pred_len * enc_in)
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model)

        self.reformer_layer = ReformerLayer(None, d_model, n_heads, bucket_size=bucket_size,
                                            n_hashes=n_hashes)

        self.encoder_layer = EncoderLayer(self.reformer_layer, d_model, d_ff, dropout=dropout, activation=activation)

        self.conv_layer = ConvLayer(d_model)
        self.norm_layer = nn.LayerNorm(d_model)
        self.encoder = Encoder(self.encoder_layer, self.conv_layer, self.norm_layer, e_layers)

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_dec = self.fc(x_dec)
        x_dec = x_dec.view(-1, self.pred_len, self.enc_in)
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.label_len:, :]], dim=1)

        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.label_len:, :]], dim=1)

        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)
        output = enc_out[:, -self.pred_len:, :]  # [B, L, D]
        output = output.squeeze(-1)
        return output

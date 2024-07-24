import logging
import os

import matplotlib.pyplot as plt
import torch.nn as nn
from omegaconf import DictConfig

from nets.autoformer import Autoformer
from nets.informer import Informer
from nets.lstm import LSTM, GRU, BiLSTM, BiGRU
from nets.reformer import Reformer
from nets.tcn import TCN_LSTM


def gen_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name

    if name == 'autoformer':
        model = Autoformer(
            seq_len=cfg.model.seq_len,
            pred_len=cfg.model.pred_len,
            label_len=cfg.model.label_len,
            moving_avg=cfg.model.moving_avg,
            enc_in=cfg.model.enc_in,
            d_model=cfg.model.d_model,
            dropout=cfg.model.dropout,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            e_layers=cfg.model.e_layers,
            d_layers=cfg.model.d_layers,
            activation=cfg.model.activation,
            c_out=cfg.model.c_out,
            factor=cfg.model.factor
        )
    elif name == 'informer':
        model = Informer(
            enc_in=cfg.model.enc_in,
            dec_in=cfg.model.dec_in,
            c_out=cfg.model.c_out,
            pred_len=cfg.model.pred_len,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            e_layers=cfg.model.e_layers,
            d_layers=cfg.model.d_layers,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            factor=cfg.model.factor,
            attn=cfg.model.attn,
            activation=cfg.model.activation,

        )
    elif name == 'reformer':
        model = Reformer(
            enc_in=cfg.model.enc_in,
            d_model=cfg.model.d_model,
            dropout=cfg.model.dropout,
            e_layers=cfg.model.e_layers,
            n_heads=cfg.model.n_heads,
            bucket_size=cfg.model.bucket_size,
            n_hashes=cfg.model.n_hashes,
            d_ff=cfg.model.d_ff,
            c_out=cfg.model.c_out,
            label_len=cfg.model.label_len,
            seq_len=cfg.model.seq_len,
            pred_len=cfg.model.pred_len,
            activation=cfg.model.activation,
        )
    elif name == 'lstm':
        model = LSTM(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            d_model=cfg.model.d_model,
            pred_len=cfg.model.pred_len,
        )
    elif name == 'bilstm':
        model = BiLSTM(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            d_model=cfg.model.d_model,
            pred_len=cfg.model.pred_len,
        )
    elif name == 'gru':
        model = GRU(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            d_model=cfg.model.d_model,
            pred_len=cfg.model.pred_len,
        )
    elif name == 'bigru':
        model = BiGRU(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            d_model=cfg.model.d_model,
            pred_len=cfg.model.pred_len,
        )
    elif name == 'tcn_lstm':
        model = TCN_LSTM(
            input_size=cfg.model.input_size,
            num_channels=cfg.model.num_channels,
            lstm_hidden_size=cfg.model.lstm_hidden_size,
            lstm_num_layers=cfg.model.lstm_num_layers,
            dropout=cfg.model.dropout,
            pred_len=cfg.model.pred_len,
            kernel_size=cfg.model.kernel_size,
        )
    else:
        raise ValueError(f'还没有提供这个模型: {name}')

    print(f"模型-{name}创建成功")
    return model


def setup_logging(run_dir):
    log_file = os.path.join(run_dir, 'train.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )


def draw_loss_curve(train_losses, run_dir):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, f'loss_curve.png'))
    plt.close()

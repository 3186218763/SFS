import logging
import os

import torch.nn as nn
from omegaconf import DictConfig

from nets.autoformer import Autoformer
from nets.lstm import LSTM, GRU, BiLSTM, BiGRU


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

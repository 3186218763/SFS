import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

from nets.dataset import TimeSeriesDataset
from utools.train_tools import setup_logging, gen_model, draw_loss_curve


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 设置日志模块
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    setup_logging(run_dir)
    logger = logging.getLogger(__name__)

    # 构建原始的dataset
    dataset = TimeSeriesDataset(cfg.data.csv_file, cfg.model.seq_len, cfg.model.pred_len)
    dataset_size = len(dataset)
    train_size = int(cfg.data.train_split * dataset_size)

    # 创建训练和验证数据集，按顺序分离
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle,
                              num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 生成加载模型
    model = gen_model(cfg)
    model = model.to(device)
    if cfg.path.load_dir is not None:
        model.load_state_dict(torch.load(cfg.path.load_dir))
        logger.info("模型参数加载成功")

    # 训练部分
    if cfg.train.is_train:
        # 设置训练器，loss_fn，迭代器
        optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate)
        loss_fn = nn.SmoothL1Loss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)

        best_loss = float('inf')
        train_losses = []

        logger.info("开始训练...")

        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                x = batch['x'].to(device)
                x_mark = batch['x_time'].to(device)
                y_mark = batch['y_time'].to(device)
                y = batch['y'].to(device)

                zeros = torch.zeros_like(y[:, cfg.model.label_len:]).to(device)

                dec_inp = torch.cat([y[:, :cfg.model.label_len], zeros], dim=1).float().to(device)

                outputs = model(x, x_mark, dec_inp, y_mark)
                loss = loss_fn(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch + 1}/{cfg.train.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.8f}")

            # 保存损失更小的模型参数
            if avg_loss < best_loss:
                best_loss = avg_loss
                logger.info(f"最小Loss是：{best_loss:.4f}")
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))

            scheduler.step()

            # 每10个epoch绘制一次损失曲线
            if (epoch + 1) % 10 == 0:
                draw_loss_curve(train_losses, run_dir)

        logger.info("训练完成。")

    # 验证部分
    logger.info("开始验证...")
    model.eval()
    all_true_values = []
    all_predicted_values = []

    # 设置测试的批次数量
    num_batches_to_test = 500 // cfg.model.pred_len

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches_to_test:
                break

            x = batch['x'].to(device)
            x_mark = batch['x_time'].to(device)
            y_mark = batch['y_time'].to(device)
            y = batch['y'].to(device)

            dec_inp = torch.zeros_like(y).to(device)

            outputs = model(x, x_mark, dec_inp, y_mark)

            # 反归一化数据
            outputs = dataset.inverse_transform(outputs.cpu().numpy())
            y = dataset.inverse_transform(y.cpu().numpy())

            # 将所有批次的数据连接起来
            all_true_values.append(y.flatten())
            all_predicted_values.append(outputs.flatten())

    # 将所有批次的数据转换为一维数组
    all_true_values = np.concatenate(all_true_values)
    all_predicted_values = np.concatenate(all_predicted_values)

    # 绘制预测和真实值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(all_true_values, label='True')
    plt.plot(all_predicted_values, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f'{cfg.model.name}: True vs. Predicted')
    plt.savefig(os.path.join(run_dir, f'{cfg.model.name}.png'))
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()

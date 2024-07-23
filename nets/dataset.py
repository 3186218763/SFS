import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from nets.autoformer import Autoformer
from nets.informer import Informer


class TimeSeriesDataset(Dataset):
    def __init__(self, station_file, seq_len, pred_len, feature_cols=None, target_col=None):
        if feature_cols is None:
            feature_cols = ['气温', '气压', '相对湿度', '风速', '日照', '地温', '降水量']
        if target_col is None:
            target_col = '平均流量'
        self.data = pd.read_csv(station_file)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        # 提取特征列和标签列
        self.feature_cols = feature_cols
        self.target_col = target_col

        self.seq_len = seq_len
        self.pred_len = pred_len

        # 提取时间特征
        df_stamp = self.data.reset_index()[['date']]
        df_stamp['year'] = df_stamp['date'].dt.year - 2010
        df_stamp['month'] = df_stamp['date'].dt.month
        df_stamp['day'] = df_stamp['date'].dt.day
        df_stamp['weekday'] = df_stamp['date'].dt.weekday

        # 初始化归一化器
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # 归一化特征和标签
        self.data[self.feature_cols] = self.feature_scaler.fit_transform(self.data[self.feature_cols])
        self.data[self.target_col] = self.target_scaler.fit_transform(self.data[[self.target_col]])

        # 保存时间特征
        self.data_stamp = df_stamp[['year', 'month', 'day', 'weekday']].values

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        x = self.data[self.feature_cols].iloc[x_start:x_end].values
        y = self.data[self.target_col].iloc[y_start:y_end].values
        x_time = self.data_stamp[x_start:x_end]
        y_time = self.data_stamp[y_start:y_end]

        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'x_time': torch.tensor(x_time, dtype=torch.float32),
            'y_time': torch.tensor(y_time, dtype=torch.float32)
        }

    def inverse_transform(self, y):
        return self.target_scaler.inverse_transform(y)


if __name__ == '__main__':

    seq_len = 100
    pred_len = 5
    batch_size = 16
    label_len = 2
    csv_file = '../data/更张.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = Informer(
    #     enc_in=7,
    #     dec_in=4,
    #     d_model=512,
    #     dropout=0.1,
    #     n_heads=8,
    #     d_ff=2048,
    #     d_layers=1,
    #     e_layers=1,
    #     factor=1.0,
    #     pred_len=pred_len,
    #     c_out=1,
    #     activation='gelu'
    #
    #
    # ).to(device)
    model = Autoformer(
        seq_len=seq_len,
        label_len=label_len,  # seq-pre
        pred_len=pred_len,
        moving_avg=25,
        enc_in=7,
        d_model=512,  # 根据需要调整
        dropout=0.05,  # 根据需要调整
        n_heads=8,
        d_ff=2048,  # 根据需要调整
        e_layers=2,
        d_layers=1,
        activation='gelu',
        c_out=1,
        factor=1.0
    ).to(device)
    # 创建数据集和数据加载器
    dataset = TimeSeriesDataset(csv_file, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 打印一些数据检查
    for batch in dataloader:
        x = batch['x'].to(device)
        x_mark = batch['x_time'].to(device)
        y_mark = batch['y_time'].to(device)
        y = batch['y'].to(device)

        zeros = torch.zeros_like(y[:, label_len:]).to(device)

        dec_inp = torch.cat([y[:, :label_len], zeros], dim=1).float().to(device)
        print(dec_inp.shape)
        out = model(x, x_mark, dec_inp, y_mark)
        print(out.shape)

        break

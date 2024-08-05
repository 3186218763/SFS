import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    epsilon = 1e-10  # 定义一个非常小的数值
    true_safe = np.where(true == 0, epsilon, true)  # 将 true 中的零值替换为 epsilon
    return np.mean(np.abs((pred - true) / true_safe))

def MSPE(pred, true):
    epsilon = 1e-10  # 定义一个非常小的数值
    true_safe = np.where(true == 0, epsilon, true)  # 将 true 中的零值替换为 epsilon
    return np.mean(np.square((pred - true) / true_safe))


def NSE(pred, true):
    """
    计算纳什效率系数NSE

    """

    true_mean = np.mean(true)

    numerator = np.sum(np.power(true - pred, 2))
    denominator = np.sum(np.power(true - true_mean, 2))

    nse = 1 - numerator / denominator

    return nse


def metric(pred, true):
    """

    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    nse = NSE(pred, true)

    return mae, mse, rmse, mape, mspe, nse

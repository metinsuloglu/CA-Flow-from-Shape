import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mean absolute error
def mae(y_pred, y, mask=None):
    y_pred, y = y_pred.numpy().squeeze(), y.numpy().squeeze()
    if mask is not None:
        mask = mask.numpy().squeeze()
        return np.mean(np.absolute(y_pred[mask!=0] - y[mask!=0]))
    else:
        return np.mean(np.absolute(y_pred - y))

# Mean squared error
def mse(y_pred, y, mask=None):
    y_pred, y = y_pred.numpy().squeeze(), y.numpy().squeeze()
    if mask is not None:
        mask = mask.numpy().squeeze()
        return np.mean(np.power((y_pred[mask!=0] - y[mask!=0]), 2))
    else:
        return np.mean(np.power((y_pred - y), 2))

# Average Pearson correlation coefficient
def avg_pearson(y_pred, y, mask=None):
    all_r = []
    y_pred, y = y_pred.numpy().squeeze(axis=-1), y.numpy().squeeze(axis=-1)
    if mask is not None: mask = mask.numpy().squeeze(axis=-1)
    for i in range(y.shape[0]):
        if mask is not None:
            all_r.append(pearsonr(y_pred[i][mask[i]!=0], y[i][mask[i]!=0])[0])
        else:
            all_r.append(pearsonr(y_pred[i], y[i])[0])
    return np.mean(all_r)

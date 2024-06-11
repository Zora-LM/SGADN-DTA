from math import sqrt
from sklearn.linear_model import LinearRegression
import numpy as np

class Evaluate:

    def _rmse(self, y, p):
        rmse = sqrt(((y - p) ** 2).mean(axis=0))
        return rmse

    def _mae(self, y, p):
        mae = (np.abs(y - p)).mean()
        return mae

    def _sd(self, y, p):
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(p, y)
        y_ = lr.predict(p)
        sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
        return sd

    def _pearson(self, y, p):
        rp = np.corrcoef(y, p)[0, 1]
        return rp

    def evaluate(self, y_true, pred):
        rmse = self._rmse(y_true, pred)
        mae = self._mae(y_true, pred)
        r = self._pearson(y_true, pred)
        sd = self._sd(y_true, pred)
        return rmse, mae, r, sd
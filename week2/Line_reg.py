import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class LinearReg:
	def __init__(self, x, y):
		self._x = x.reshape(-1, 1)
		self._y = y.reshape(-1, 1)
		
		self.reg = LinearRegression(fit_intercept=True, n_jobs=1).fit(self._x, self._y)
		self.w1 = self.reg.coef_
		self.w0 = self.reg.intercept_
	
	def y(self, x):
		_x = x.reshape(-1, 1)
		return self.reg.predict(_x)
		
	def mse(self, y, y_pred):
		try:
			mse = mean_squared_error(y, y_pred)
			return mse
		except:
			_y_pred = y_pred.reshape(-1, 1)
			mse = mean_squared_error(y, _y_pred)
			return mse

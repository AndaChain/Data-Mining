import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

class LinearReg:
	def __init__(self, x, y, degree=1):
		self._x = x.reshape(-1, 1)
		self._y = y.reshape(-1, 1)
		self._degree = degree
		
		poly = PolynomialFeatures(degree=self._degree, include_bias=False)
		poly_features = poly.fit_transform(self._x) # as x
		
		self.reg = LinearRegression().fit(poly_features, self._y)
		self.w1 = self.reg.coef_
		self.w0 = self.reg.intercept_
	
	def y(self, x):
		_x = x.reshape(-1, 1)
		poly = PolynomialFeatures(degree=self._degree, include_bias=False)
		poly_features = poly.fit_transform(_x) # as x
		
		return self.reg.predict(poly_features)
		
	def mse(self, y, y_pred):
		try:
			mse = mean_squared_error(y, y_pred)
			return mse
		except:
			_y_pred = y_pred.reshape(-1, 1)
			mse = mean_squared_error(y, _y_pred)
			return mse

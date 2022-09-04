import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
#from Line_reg import LinearReg as line_reg
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
random.seed(1)

class model:
	def __init__(self):
		self.n = 10000 # n is show number of model
		self.N = 2 # N is show number of sample
		self.posible_x = 2000
		self.mode = "non-constant"
		self.x_real = np.linspace(-1,1, num=self.posible_x)
		self.fun = lambda x : np.sin(x*np.pi)
		self.org = False
	
	def f(self, x): # input function
		return self.fun(x)
	
	def set_n_model(self, n):
		self.n = n

	def set_N_sample(self, N):
		self.N = N

	def set_posible_x(self, posible_x):
		self.posible_x = posible_x
		self.x_real = np.linspace(-1,1, num=self.posible_x)

	def random_input(self, N, start, stop):# random input function
		x = np.array([])
		for n in range(N): # N is show number of sample
			x = np.append(x, [random.uniform(start, stop)])
		y = self.f(x)
		return x,y
	
	def create_model(self, x, y):# show model
		#return LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)
		#return Ridge(alpha=10).fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)
		return Lasso(alpha=0.35).fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)

	def get_model_avg(self, models): # show model average
		sum_w1 = 0
		sum_w0 = 0
		for g in models:
			w1 = g.coef_
			w0 = g.intercept_
			sum_w1 += w1
			sum_w0 += w0
		return self.x_real*sum_w1/len(models) + sum_w0/len(models)

	def get_model(self): # show bias
		models = np.array([])
		for i in range(self.n): # n is show number of model
			x,y = self.random_input(self.N, -1, 1)
			_model = self.create_model(x,y)
			
			print( mean_squared_error(y, _model.predict(x.reshape(-1, 1))) )
			
			models = np.append(models, [_model])
			plt.plot( self.x_real, _model.predict(self.x_real.reshape(-1, 1)), "-", c='k' )
		return models

	def main(self):
		self.set_n_model(10)
		arr_N = np.arange(2,21,1)
		for N in arr_N:
			self.set_N_sample(N)
			models = self.get_model()
		
		g_aveg = self.get_model_avg(models)
		plt.plot( self.x_real, self.fun(self.x_real.reshape(-1, 1)), "-", linewidth=7.0)
		plt.plot( self.x_real, g_aveg, "-")
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		
		plt.show()
		
if __name__ == '__main__':
	obj = model()
	obj.main()

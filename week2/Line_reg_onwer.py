import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def show(x,y,format_):
	plt.rcParams['figure.figsize'] = (12.0, 9.0)
	plt.plot(x, y, format_)
	plt.show()

def data(name):
	data = pd.read_csv(name)
	x = data.iloc[:, 0].to_numpy()
	y = data.iloc[:, 1].to_numpy()
	return x,y
"""
class model:
	def __init__(self, w1=1, w0=0):
		self.w1 = w1
		self.w0 = w0
	
	def y(self, x):
		return self.w0 + self.w1*x

	def set_w1(self, w1):
		self.w1 = w1

	def set_w0(self, w0):
		self.w0 = w0

def MSE(model, x, y):
	sum_out = 0
	N = len(y)
	
	for i in range(N):
		sum_out += ( model.y(x[i]) - y[i] )**2
	
	#sum_out = sum( ( y - model.y(x) )**2 )
	return sum_out/(N)

def d_MSE_w1(model, x, y):
	sum_out = model.w1
	N = len(y)
	#_min = np.array([])
	
	for i in range(N):
		_t = ( model.y(x[i])-y[i] )*x[i]
		sum_out += (2/N)*(_t)
		#_min = np.append(_min, [_t])
	
	#sum_out = sum( ( model.y(x) - y )*x )
	return sum_out #np.min(_min)

def d_MSE_w0(model, x, y):
	sum_out = model.w0
	N = len(y)
	#_min = np.array([])
	
	for i in range(N):
		_t = model.y(x[i]) - y[i]
		sum_out += (2/N)*(_t)
		#_min = np.append(_min, [_t])
	
	#sum_out = sum( ( model.y(x) - y ) )
	return sum_out #np.min(_min)
	

def update_weights(model, x, y, learning):
	w1_deriv = 0
	w0_deriv = 0
	w1 = model.w1
	w0 = model.w0
	N = len(x)

	# Calculate partial derivatives
	# -2x(y - (mx + b))
	w1_deriv = sum(  -2*x*(y - g.y(x))  )

	# -2(y - (mx + b))
	w0_deriv = sum(  -2*(y-g.y(x))  )

    # We subtract because the derivatives point in direction of steepest ascent
	w1 -= (w1_deriv / N) * learning
	w0 -= (w0_deriv / N) * learning

	model.set_w1(w1)
	model.set_w0(w0)
"""

if __name__ == '__main__':
	name = 'HeightWeight.csv'
	#name = 'RocketPropellant.csv'
	#round_gradient = 1000
	x_real = np.linspace(-200,200, 10000)
	x,y = data(name)
	plt.plot(x, y, "o")
	
	_x = x.reshape(-1, 1)
	_y = y.reshape(-1, 1)
	reg = LinearRegression(fit_intercept=True, n_jobs=1).fit(_x, _y)
	y_pred = reg.predict(_x) # training set
	plt.plot(_x, y_pred, "-") # training set
	
	coef = reg.coef_
	intercept = reg.intercept_
	mse =mean_squared_error(_y, y_pred)
	
	print(y_pred)
	print(coef)
	print(intercept)
	print(np.sqrt(mse))
	
	plt.show()
	
	"""
	learning = 0.000001
	w1 = 0
	w0 = 0
	g = model(w1, w0) ##
	old_mse = MSE(g, x, y) ##
	
	y_space = np.arange(-np.max(y),np.max(y),0.001)
	x_space = np.arange(-np.max(x),np.max(x),0.001)
	print(len(x_space))
	for _w0 in y_space:
		_g = model(w1, _w0)
		_mse = MSE(_g, x, y)
		if(_mse < old_mse):
			w0 = _w0
	
	g = model(w1, w0) ##
	old_mse = MSE(g, x, y) ##

	for _w1 in x_space:
		_g = model(_w1, w0)
		_mse = MSE(_g, x, y)
		if(_mse < old_mse):
			w1 = _w1
	
	g = model(w1, w0) ##
	old_mse = MSE(g, x, y) ##

	for _ in range(round_gradient):
		min_w1 = d_MSE_w1(g, x, y)
		min_w0 = d_MSE_w0(g, x, y)
		
		diff_w1 = min_w1*learning
		diff_w0 = min_w0*learning
		
		w1 = w1 - diff_w1
		w0 = w0 - diff_w0
		
		g.set_w1(w1)
		g.set_w0(w0)
	
	mse = MSE(g, x, y)
	print("**************************")
	print(np.sqrt(mse),end="   ")
	print(f"y = x{w1}+{w0}")
		
	plt.plot(x,y, "o")
	plt.plot(x_real, g.y(x_real), "-")
	plt.show()
	

	for r in range(round_gradient):
		#print(g.w1,g.w0)
		mse = MSE(g,x,y)
		print(mse)
		
		diff_w1 = -learning*d_MSE_w1(g,x,y)
		w1 += diff_w1
		g.set_w1(w1)
		
		diff_w0 = -learning*d_MSE_w0(g,x,y)
		w0 += diff_w0
		g.set_w0(w0)
	"""

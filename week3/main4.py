from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)

def random_input(N, start, stop):# random input function
	x = np.array([])
	x = np.linspace(start, stop, N)
	
	f = lambda x: np.sin(np.pi*x) #10*random.uniform(0.5, stop)*x+10*random.uniform(start, stop)
	y = f(x)
	for n in range(N): # N is show number of sample
		y[n] = y[n]+random.uniform(0, 0.001)
	return x,y

def x_y_al():
	########Data#########
	x, y = random_input(1000, -1, 1)
	plt.plot(x, y, "o")
	plt.show()
	########Data#########

	########Ridge&Lasso#########
	arr_legend = np.array([])


	step_w1_times = 2
	Alpha = np.linspace(0, 900, 10)
	w1_r = 1
	w1_l = 1

	for alpha in Alpha:
		clf_R = Ridge(alpha=alpha)
		clf_R.fit(x.reshape(-1, 1), y.reshape(-1, 1))
		########
		clf_L = Lasso(alpha=alpha)
		clf_L.fit(x.reshape(-1, 1), y.reshape(-1, 1))
		
		w1_R = clf_R.coef_
		w0_R = clf_R.intercept_
		y_predict_R = clf_R.predict(x.reshape(-1, 1))
		
		arr_se_R = np.array([])
		arr_se_L = np.array([])
		arr_w1_R = np.array([])
		arr_w1_L = np.array([])
		
		arr_w1 = np.arange(-1,1,0.01)
		
		for _w1 in arr_w1:
			se_R = np.sum( (y-y_predict_R)**2 ) # mean_squared_error(y,y_predict_R)*len(y_predict_R)
			arr_se_R = np.append(arr_se_R, [se_R+alpha*(_w1**2)]) #
			arr_w1_R = np.append(arr_w1_R, [_w1])
		
		_x_R = x.reshape(-1, 1)
		########
		w1_L = clf_L.coef_
		w0_L = clf_L.intercept_
		y_predict_L = clf_L.predict(x.reshape(-1, 1))
		se_L = np.sum( (y_predict_L-y)**2 )
		arr_se_L = np.append(arr_se_L, [se_L]) #+alpha*np.absolute(w1_L)
		arr_w1_L = np.append(arr_w1_L, [w1_L])
		_x_L = x.reshape(-1, 1)
		
		
		if(w1_r/w1_R > step_w1_times or alpha == Alpha[-1]):
			w1_r = w1_R
			#plt.plot(arr_w1_R, arr_se_R, "-")
			plt.plot(_x_R, y_predict_R, "--")
			arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha)+", w1:"+str(np.round(w1_R[0][0], 3))])
		"""
		if(w1_l/w1_L > step_w1_times or alpha == Alpha[-1]):
			w1_l = w1_L
			plt.plot(_x_L, y_predict_L, "--")
			arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha/10000)+", w1:"+str(np.round(w1_L[0], 3))])
		"""
	########Ridge&Lasso#########

	########Linear#########
	Lge = LinearRegression()
	Lge.fit(x.reshape(-1, 1), y.reshape(-1, 1))

	w1 = Lge.coef_
	w0 = Lge.intercept_
	y_predict = Lge.predict(x.reshape(-1, 1))
	_x = x.reshape(-1, 1)
	plt.plot(_x, y_predict, "-")
	arr_legend = np.append(arr_legend, ["LinearReg"])
	########Linear#########

	plt.legend(arr_legend, loc='best')
	plt.xlabel('X axis')
	plt.ylabel('Y axis')
	plt.title('LinearRegression & Ridge')

	plt.show()

x_y_al()

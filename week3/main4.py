from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)

########Data#########
n_samples, n_features = 1000, 1
rng = np.random.RandomState(1)
f = lambda x : random.random()*x+ random.randrange(-10,10)*random.random()
x = rng.randn(n_samples, n_features)
x_noisy = np.array([i+2*random.random() for i in x])
y = f(x_noisy) #rng.randn(n_samples)
#plt.plot(x, y, "o")
########Data#########

########Ridge&Lasso#########
arr_legend = np.array([])


step_w1_times = 5
Alpha = 60000
w1_r = 1
w1_l = 1

for alpha in range(0,Alpha+500,500):
	clf_R = Ridge(alpha=alpha)
	clf_R.fit(x_noisy, y)
	########
	clf_L = Lasso(alpha=alpha/10000)
	clf_L.fit(x_noisy, y)
	
	w1_R = clf_R.coef_
	w0_R = clf_R.intercept_
	y_predict_R = clf_R.predict(x_noisy)
	
	
	arr_se_R = np.array([])
	arr_se_L = np.array([])
	arr_w1_R = np.array([])
	arr_w1_L = np.array([])
	for _w1 in range(-100,100):
		se_R = np.sum( (y-(w0_R+_w1*w1_R))**2 ) # mean_squared_error(y,y_predict_R)*len(y_predict_R)
		arr_se_R = np.append(arr_se_R, [se_R+alpha*(_w1**2)]) #
		arr_w1_R = np.append(arr_w1_R, [_w1])
	
	_x_R = x_noisy.reshape(-1, 1)
	########
	w1_L = clf_L.coef_
	w0_L = clf_L.intercept_
	y_predict_L = clf_L.predict(x_noisy)
	se_L = np.sum( (y_predict_L-y)**2 )
	arr_se_L = np.append(arr_se_L, [se_L]) #+alpha*np.absolute(w1_L)
	arr_w1_L = np.append(arr_w1_L, [w1_L])
	_x_L = x_noisy.reshape(-1, 1)
	

	if(w1_r/w1_R > step_w1_times or alpha == Alpha):
		w1_r = w1_R
		print(w1_R)
		plt.plot(arr_w1_R, arr_se_R, "-")
		#plt.plot(_x_L, y_predict_L, "--")
		#arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha)+", w1:"+str(np.round(w1_R[0][0], 3))])
	
	"""
	if(w1_l/w1_L > step_w1_times or alpha == Alpha):
		w1_l = w1_L
		plt.plot(_x_L, y_predict_L, "--")
		arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha/10000)+", w1:"+str(np.round(w1_L[0], 3))])
	"""

########Ridge&Lasso#########

########Linear#########
Lge = LinearRegression()
Lge.fit(x_noisy, y)

w1 = Lge.coef_
w0 = Lge.intercept_
y_predict = Lge.predict(x_noisy)
_x = x_noisy.reshape(-1, 1)
#plt.plot(_x, y_predict, "-")
#arr_legend = np.append(arr_legend, ["LinearReg"])
########Linear#########

#plt.legend(arr_legend, loc='best')
#plt.xlabel('X axis')
#plt.ylabel('Y axis')
#plt.title('LinearRegression & Lasso')

plt.show()

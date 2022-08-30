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
	for n in range(N): # N is show number of sample
		x = np.append(x, [random.uniform(start, stop)])
	f = lambda x : np.sin(pi*x)
	y = f(x)
	return x,y

def cross_vali(x,y,seed=1,n=10,degree=1):
	kf = KFold(n_splits=n, random_state=seed, shuffle=True)
	Mean_kf = 0
	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		reg = line_reg(x_train,y_train,degree)
		w1 = reg.w1
		w0 = reg.w0
		y_pred = reg.y(x_test)
		mse = reg.mse(y_test, y_pred)
		
		Mean_kf += np.sqrt(mse)
	
	return Mean_kf/n

x, y = random_input(2, -1, 1)


"""
########Data#########
n_samples, n_features = 1000, 1
rng = np.random.RandomState(1)

f = lambda x : random.random()*x+ random.randrange(-10,10)
x = rng.randn(n_samples, n_features)
x_noisy = np.array([i+2*random.random() for i in x])
_y = f(x_noisy) #rng.randn(n_samples)
y = np.array([i+1.5*random.random() for i in _y])

plt.plot(x, y, "o")
plt.show()
########Data#########

########Ridge&Lasso#########
arr_legend = np.array([])


step_w1_times = 2
Alpha = 10000
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
	
	arr_w1 = np.arange(-1,1,0.01)
	
	for _w1 in arr_w1:
		se_R = np.sum( (y-y_predict_R)**2 ) # mean_squared_error(y,y_predict_R)*len(y_predict_R)
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
		#plt.plot(_x_R, y_predict_R, "--")
		arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha)+", w1:"+str(np.round(w1_R[0][0], 3))])
	
	if(w1_l/w1_L > step_w1_times or alpha == Alpha):
		w1_l = w1_L
		plt.plot(_x_L, y_predict_L, "--")
		arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha/10000)+", w1:"+str(np.round(w1_L[0], 3))])
	
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
"""

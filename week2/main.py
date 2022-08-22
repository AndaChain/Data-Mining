import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

from Line_reg import LinearReg as line_reg
	
def data(name):
	data = pd.read_csv(name)
	x = data.iloc[:, 0].to_numpy()
	y = data.iloc[:, 1].to_numpy()
	return x,y
	
def training_set(x,y):
	reg = line_reg(x,y)
	
	y_pred = reg.y(x)
	w1 = reg.w1
	w0 = reg.w0
	mse = reg.mse(y, y_pred)
	
	#print(y_pred)
	#print(w1)
	#print(w0)
	#print(np.sqrt(mse))
	
	#plt.plot(x, y_pred, "-") # training set
	
	return np.sqrt(mse)

def holdout(x,y,seed,test_size):
	x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed)
	#print(x_train,y_train)
	#print(x_test,y_test)
	
	reg = line_reg(x_train,y_train)
	
	y_pred = reg.y(x_test)
	w1 = reg.w1
	w0 = reg.w0
	mse = reg.mse(y_test, y_pred)
	
	#print(y_pred)
	#print(w1)
	#print(w0)
	#print(np.sqrt(mse))
	#plt.plot(x_test, y_pred, "-") # holdout

	return np.sqrt(mse)
	
def cross_vali(x,y,seed,n):
	kf = KFold(n_splits=n, random_state=seed, shuffle=True)
	Mean_kf = 0
	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		reg = line_reg(x_train,y_train)
		
		y_pred = reg.y(x_test)
		w1 = reg.w1
		w0 = reg.w0
		mse = reg.mse(y_test, y_pred)
		
		#print(y_pred)
		#print(w1)
		#print(w0)
		#print(np.sqrt(mse))
		Mean_kf += np.sqrt(mse)
		#plt.plot(x_test, y_pred, "-") # holdout
	
	return Mean_kf/n

if __name__ == '__main__':
	name = 'HeightWeight100.csv'
	#name = 'RocketPropellant.csv'
	#round_gradient = 1000
	x_real = np.linspace(-200,200, 10000)
	x,y = data(name)
	#plt.plot(x, y, "o")
	
	arr_h = []
	arr_c = []
	
	arr_aveg_h = []
	arr_std_h = []
	
	arr_aveg_c = []
	arr_std_c = []
	
	#set_h = {}
	#set_c = {}
	
	rmse_t = training_set(x,y) # training set
	arr_test_size = np.arange(0.1, 1, 0.1)
	print("holdout")
	for test_size in arr_test_size: #y
		arr_h = []
		for seed in range(1,101): #x
			rmse_h = holdout(x,y,seed,test_size) # holdout
			arr_h.append(rmse_h)
		arr_aveg_h.append( sum(arr_h)/len(arr_h) )
		arr_std_h.append( np.std(arr_h) )
		#print( round(test_size,2), round(sum(arr_h)/len(arr_h),3), np.std(arr_h) )
		
	plt.plot(arr_test_size, arr_aveg_h, "-")
	plt.xlabel('Split ratio of Data to Testing')
	plt.ylabel('RMSE')
	plt.title('X:Split ratio & Y:RMSE')
	plt.show()
	
	plt.plot(arr_test_size, arr_std_h, "--")
	plt.xlabel('Split ratio of Data to Testing')
	plt.ylabel('STD')
	plt.title('X:Split ratio & Y:STD')
	plt.show()
	
	_kf = np.arange(2, 11, 1)
	print("cross validation")
	for k in _kf: #y
		arr_c = []
		for seed in range(14501,14601): #x
			rmse_c = cross_vali(x,y,seed,k) # cross_validation
			arr_c.append(rmse_c)
		arr_aveg_c.append( sum(arr_c)/len(arr_c) )
		arr_std_c.append( np.std(arr_c) )
		#print( round(k,2), round(sum(arr_c)/len(arr_c),3), np.std(arr_h) )
	
	plt.plot(_kf, arr_aveg_c, "-")
	plt.xlabel('KFold')
	plt.ylabel('RMSE')
	plt.title('X:KFold & Y:RMSE')
	plt.show()
	
	plt.plot(_kf, arr_std_c, "--")
	plt.xlabel('KFold')
	plt.ylabel('STD')
	plt.title('X:KFold & Y:STD')
	plt.show()

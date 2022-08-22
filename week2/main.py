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
	
def training_set(x,y,seed=1,test_size=10,random=False):
	if(random):
		x_, x, y_, y = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed)
	reg = line_reg(x,y)
	y_pred = reg.y(x)
	w1 = reg.w1
	w0 = reg.w0
	mse = reg.mse(y, y_pred)
	
	return np.sqrt(mse)

def holdout(x,y,seed,test_size):
	x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed)
	reg = line_reg(x_train,y_train)
	y_pred = reg.y(x_test)
	mse = reg.mse(y_test, y_pred)
	
	return np.sqrt(mse)
	
def cross_vali(x,y,seed,n):
	kf = KFold(n_splits=n, random_state=seed, shuffle=True)
	Mean_kf = 0
	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		reg = line_reg(x_train,y_train)
		y_pred = reg.y(x_test)
		mse = reg.mse(y_test, y_pred)
		
		Mean_kf += np.sqrt(mse)
	
	return Mean_kf/n

if __name__ == '__main__':
	name = 'HeightWeight.csv'
	#name = 'RocketPropellant.csv'
	x_real = np.linspace(-200,200, 10000)
	x,y = data(name)
	#plt.plot(x, y, "o")
	
	seed_N = 10000
	
	arr_h = []
	arr_c = []
	
	arr_aveg_h = []
	arr_std_h = []
	
	arr_aveg_c = []
	arr_std_c = []
	
	# ------------------holdout-------------------
	arr_test_size = np.arange(0.1, 1, 0.1)
	print("------------------holdout------------------")
	for test_size in arr_test_size: #y
		arr_h = []
		for seed in range(seed_N): #x
			rmse_h = holdout(x,y,seed+1,test_size) # holdout
			arr_h.append(rmse_h)
		aveg = sum(arr_h)/len(arr_h)
		std = np.std(arr_h)
		
		arr_aveg_h.append( aveg )
		arr_std_h.append( std )
		
		print( f"Split ratio:{round(test_size, 3)}     RMSE:{round(aveg, 3)}     STD:{std}" )
	
	# ------------------ plot holdout-------------------
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
	
	# ------------------cross validation-------------------
	_kf = np.arange(2, 11, 1)
	print("------------------cross validation------------------")
	for k in _kf: #y
		arr_c = []
		for seed in range(seed_N): #x
			rmse_c = cross_vali(x,y,seed+1,k) # cross_validation
			arr_c.append(rmse_c)
		aveg = sum(arr_c)/len(arr_c)
		std = np.std(arr_c)
		
		arr_aveg_c.append( aveg )
		arr_std_c.append( std )
		
		print( f"KFold:{round(k, 3)}     RMSE:{round(aveg, 3)}     STD:{std}" )
	
	# ------------------ plot cross validation-------------------
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
	
	# ------------------matric & training set rmse-------------------
	sum_rmse_t = 0
	sum_rmse_h = 0
	sum_rmse_c = 0
	N = 0
	for seed in range(seed_N):
		sum_rmse_t += training_set(x,y,seed+1,0.002,True)
		sum_rmse_h += holdout(x,y,seed+1,0.002)
		sum_rmse_c += cross_vali(x,y,seed+1,500)
		N += 1
	rmse_t = training_set(x,y) # training set NO Random
	print(f"\nrmse training set(10000): {rmse_t}")
	print(f"rmse training set(20*random): {sum_rmse_t/N}")
	print(f"rmse holdout: {sum_rmse_h/N}")
	print(f"rmse cross validation: {sum_rmse_c/N}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from Line_reg import LinearReg as line_reg

import random
random.seed(1)

def data(name):
	data = pd.read_csv(name)
	for i in range(0,len(data.columns)-1):
		x = data.iloc[:, i].to_numpy()
		for j in range(len(x)):
			if(type(x[j])==type("")):
				if(x[j] == "Male"):
					x[j] = 1
				else:
					x[j] = 0
	
	#y = data.iloc[:, 2].to_numpy()
	x2 = data.iloc[:, 1].to_numpy()
	x1= data.iloc[:, 0].to_numpy()
	
	return x1,x2
	
def training_set(x,y,seed=1,test_size=10,random=False,degree=1):
	
	if(random):
		x_, x, y_, y = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed)
	
	reg = line_reg(x,y,degree)
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
		#plt.plot(x_test, y_pred, "o")
		
		mse = reg.mse(y_test, y_pred)
		
		Mean_kf += np.sqrt(mse)
	return Mean_kf/n

def f(x):
	return np.sin(np.pi*x) # from data, they are sin(pi)

def main():
	name = 'HeightWeight100.csv'
	#name = 'RocketPropellant.csv'
	x_real = np.linspace(-200,200, 10000)
	x,y = data(name)
	#plt.plot(x, y, "o")
	
	seed_N = 1000
	
	arr_h = []
	arr_c = []
	
	arr_aveg_h = []
	arr_std_h = []
	
	arr_aveg_c = []
	arr_std_c = []
	
	# 
	_rmse_t = training_set(x,y)
	
	# ------------------holdout-------------------
	arr_test_size = np.arange(0.05, 1, 0.05)
	print("------------------holdout------------------")
	for test_size in arr_test_size:
		arr_h = []
		for seed in range(seed_N):
			rmse_h = holdout(x,y,seed+1,test_size) # holdout
			arr_h.append(rmse_h)
		aveg = sum(arr_h)/len(arr_h)
		std = np.std(arr_h)
		
		arr_aveg_h.append( aveg )
		arr_std_h.append( std )
		
		print( f"Split ratio:{round(test_size, 3)}     RMSE:{round(aveg, 3)}     STD:{std}" )
	
	# ------------------ plot holdout-------------------
	plt.plot(arr_test_size, arr_aveg_h, "-")
	plt.plot(arr_test_size, len(arr_aveg_h)*[_rmse_t], "--")
	plt.xlabel('Split ratio of Data to Testing')
	plt.ylabel('RMSE')
	plt.title('X:Split ratio & Y:RMSE')
	plt.show()
	
	plt.plot(arr_test_size, arr_std_h, "-")
	plt.xlabel('Split ratio of Data to Testing')
	plt.ylabel('STD')
	plt.title('X:Split ratio & Y:STD')
	plt.show()
	
	# ------------------cross validation-------------------
	_kf = np.arange(2, 30, 2)
	print("------------------cross validation------------------")
	for k in _kf:
		arr_c = []
		for seed in range(seed_N):
			rmse_c = cross_vali(x,y,seed+1,k) # cross_validation
			arr_c.append(rmse_c)
		aveg = sum(arr_c)/len(arr_c)
		std = np.std(arr_c)
		
		arr_aveg_c.append( aveg )
		arr_std_c.append( std )
		
		print( f"KFold:{round(k, 3)}     RMSE:{round(aveg, 3)}     STD:{std}" )
	
	# ------------------ plot cross validation-------------------
	plt.plot(_kf, arr_aveg_c, "-")
	plt.plot(_kf, len(arr_aveg_c)*[_rmse_t], "--")
	plt.xlabel('KFold')
	plt.ylabel('RMSE')
	plt.title('X:KFold & Y:RMSE')
	plt.show()
	
	plt.plot(_kf, arr_std_c, "-")
	plt.xlabel('KFold')
	plt.ylabel('STD')
	plt.title('X:KFold & Y:STD')
	plt.show()
	
	# ------------------Matric & Sampling Size-------------------
	name = 'HeightWeight.csv'
	x,y = data(name)
	arr_rmse_t = []
	arr_rmse_h = []
	arr_rmse_c = []
	
	arr_std_t = []
	arr_std_h = []
	arr_std_c = []
	
	att_n = []
	
	for _n in range(10,50,10):
		sampling_ratio = 1-(_n/len(x))
		att_n.append(_n)	
		
		arr_t = []
		arr_h = []
		arr_c = []
		for seed in range(seed_N):
			x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=sampling_ratio, random_state=seed)
			rmse_t = training_set(x_train,y_train)
			rmse_h = holdout(x_train,y_train,seed+1,0.9)
			rmse_c = cross_vali(x_train,y_train,seed+1,10)
			
			arr_t.append(rmse_t)
			arr_h.append(rmse_h)
			arr_c.append(rmse_c)
		
		aveg_t = sum(arr_t)/len(arr_t)
		std_t = np.std(arr_t)
		aveg_h = sum(arr_h)/len(arr_h)
		std_h = np.std(arr_h)
		aveg_c = sum(arr_c)/len(arr_c)
		std_c = np.std(arr_c)
		
		arr_rmse_t.append( aveg_t )
		arr_std_t.append( std_t )
		arr_rmse_h.append( aveg_h )
		arr_std_h.append( std_h )
		arr_rmse_c.append( aveg_c )
		arr_std_c.append( std_c )

	# ------------------plot Matric & Sampling Size-------------------
	plt.plot(att_n, arr_rmse_t, "-")
	plt.plot(att_n, arr_rmse_h, "-")
	plt.plot(att_n, arr_rmse_c, "-")
	plt.legend(["training set",'holdout 90%','cross validation 10 fold'], loc='best')
	plt.xlabel('Sampling Size')
	plt.ylabel('RMSE')
	plt.title('Accuracy Test')
	plt.show()

	plt.plot(att_n, arr_std_t, "-")
	plt.plot(att_n, arr_std_h, "-")
	plt.plot(att_n, arr_std_c, "-")
	plt.legend(["training set",'holdout 90%','cross validation 10 fold'], loc='best')
	plt.xlabel('Sampling Size')
	plt.ylabel('STD')
	plt.title('Precision Test')
	plt.show()

def degreeAndError():
	sample = 20
	tag = "noisy"
	#name = "sin_"+tag+"_"+str(sample)+"sample.csv"
	#data = pd.read_csv(name)
	n = 80+1 #len(data.iloc[:,].to_numpy()[0])
	
	x_1 = np.arange( -1,1,0.2/(int(sample/10)) ) #data.iloc[:, 0].to_numpy()#np.arange( -1,1,0.2/(int(sample/10)) )
	y = f(x_1) #data.iloc[:, 8].to_numpy()#f(x_1)
	if(tag == "noisy"):
			ran_f = np.array([i+random.uniform(0,0.5) for i in f(x_1)])
			y = ran_f#data.iloc[:, -1].to_numpy() #f(x_1)
	#plt.plot(x_1, y,"o-")
	#plt.plot(x_1,data.iloc[:, 8].to_numpy(),"o--")
	
	#YY = mean_squared_error(y,data.iloc[:, 8].to_numpy())
	#plt.plot(x_1,[YY]*sample,"-")
	#plt.show()
	#print(YY)
	
	Y1 = np.array([])
	Y2 = np.array([])
	X = np.array([i for i in range(1,n)])
	X_min = np.max(X)
	Y_min = np.max(y)
	
	#arr = [cross_vali(x_1, y, degree=2, seed=s) for s in range(1,1000)]
	#print( sum(arr)/len(arr) )

	for d in X:
		arr = [cross_vali(x_1, y, degree=d, seed=s) for s in range(1,501)] # number of seed 500, [1,501)
		_temp = sum(arr)/len(arr) #cross_vali(x_1, y, degree=d)
		print(d,_temp)
		Y1 = np.append(Y1, [ training_set(x_1, y, degree=d) ])
		Y2 = np.append(Y2, [ _temp ])
		
		if(Y_min > _temp):
			Y_min = _temp
			X_min = d
		#plt.title("degree, rmse: "+str(d)+", "+str(_temp))
		#plt.show()
	############ Real graph ###############
	"""
	for d in range(X_min, X_min+50, 5):
		print(d)
		reg = line_reg(x_1,y,d)
		y_pred = reg.y(x_1)
		plt.plot(x_1, y,"o")
		plt.plot(x_1, y_pred, "-")
		plt.show()
	"""
	############ Real graph ###############
	
	plt.plot(X,Y1,"-")
	plt.plot(X,Y2,"-")
	plt.plot(X_min,Y_min,"o")
	plt.annotate("("+str(X_min)+" , "+str(np.round(Y_min,3))+")",(X_min,Y_min))
	plt.legend(["training set",'cross validation 10 fold', 'Ok! Degree'], loc='best')
	plt.xlabel('Degree of Fit')
	plt.ylabel('RMSE')
	plt.title('training set & cross validation by '+str(sample)+' sample '+"("+tag+")")
	plt.savefig('1.jpg')
	plt.show()
	
def sampleAndError():
	degree = 30 #len(data.iloc[:,].to_numpy()[0])
	sample = 50
	start_x = -1
	end_x = 1
	tag = "noisy"
	#name = "sin_"+tag+"_"+str(sample)+"sample.csv"
	#data = pd.read_csv(name)
	
	Y1 = np.array([])
	Y2 = np.array([])
	X = np.array([i for i in range(10,sample+1)])
	
	####temp####
	x_1 = np.arange( start_x,end_x,0.2/(int(sample/10)) ) # data.iloc[:, 0].to_numpy()
	y =  f(x_1)
	####temp####
	
	X_min = np.max(X)
	Y_min = np.max(y)	
	
	for s in X:
		diff = 0.2/(s/10)
		x_1 = np.array([])
		for i in range(s):
			x_1 = np.append(x_1, [start_x+(i*diff)]) #np.arange( -1,1,0.2/diff ) # data.iloc[:, 0].to_numpy()
		y =  f(x_1)
			
		if(tag == "noisy"):
			ran_f = np.array([i+random.uniform(-1,1) for i in y])
			y = ran_f#data.iloc[:, -1].to_numpy() #f(x_1)
		
		
		arr = [cross_vali(x_1, y, degree=degree, seed=s) for s in range(1,501)] # number of seed 500, [1,501)
		_temp = sum(arr)/len(arr) #cross_vali(x_1, y, degree=d)
		print(s,_temp)
		#_temp = cross_vali(x_1, y, degree=degree)
		Y1 = np.append(Y1, [ training_set(x_1, y, degree=degree) ])
		Y2 = np.append(Y2, [ _temp ])
		
		if(Y_min > _temp):
			Y_min = _temp
			X_min = s
		
		#plt.title("sample, rmse: "+str(s)+", "+str(_temp))
		#plt.show()
	plt.plot(X,Y1,"-")
	plt.plot(X,Y2,"-")
	plt.plot(X_min,Y_min,"o")
	plt.annotate("("+str(X_min)+" , "+str(np.round(Y_min,3))+")",(X_min,Y_min))
	plt.legend(["training set",'cross validation 10 fold', 'Ok! Sample'], loc='best')
	plt.xlabel('Number of Sample')
	plt.ylabel('RMSE')
	plt.title('training set & cross validation by '+str(degree)+' degree'+"("+tag+")")
	plt.savefig('2.jpg')
	plt.show()

if __name__ == '__main__':
	#degreeAndError()
	sampleAndError()

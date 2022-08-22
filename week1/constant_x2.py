import numpy as np
import matplotlib.pyplot as plt

def set_graph(): # set display graph function
	plt.ylim(-1, 1)
	plt.xlim(-1, 1)
	plt.grid()
	plt.legend(["real",'data'], loc='best')

def f(x): # input function
	return x**2

def real_func(start, stop, N): # real function
	x_real = np.linspace(start,stop, num=N)
	y_real = f(x_real)
	plt.plot(x_real,y_real)

def bias2(g_bar, fun):
	_bias2 = (g_bar-fun)**2
	return sum(_bias2)/len(_bias2)

def make_model(start, stop, N): # make avg. model
	sum_y = 0
	x1 = 0
	x2 = 0
	sum_intercept = 0
	x_vals = np.linspace(start,stop, num=N)
	y_arr = []
	for i in range(N):
		x1 = x_vals[i]
		sum_y2 = 0
		for j in range(N):
			x2 = x_vals[j]
			_y = (f(x1)+f(x2))/2
			sum_y += _y
			sum_y2 += _y
		y_arr.append(sum_y2/N)
		
	
	g_avg = sum_y/(N*N)
	bias2_avg = bias2(g_avg, f(x_vals))
	
	x1 = 0
	x2 = 0
	sum_y = 0
	for i in range(N):
		x1 = x_vals[i]
		for j in range(N):
			x2 = x_vals[j]
			sum_y += ((f(x1)+f(x2))/2)**2
	var = sum_y/(N*N)
	
	print(f"bias2: {bias2_avg}")
	print(f"var: {var}")
	print(f"var+bias2_avg: {var+bias2_avg}")
	
	plt.plot(x_vals, len(x_vals)*[g_avg], '--')
	Ein = sum(  ( np.array(y_arr)-f(x_vals) )**2  )/(2*N)
	Eout = var+bias2_avg
	return Ein, Eout
	
if __name__ == '__main__':
	N = 1000 # จำนวนรอบ
	start = -1 # x น้อยสุด
	stop = 1 # x มากสุด
	real_func(start, stop, N)
	Ein, Eout = make_model(start, stop, N)
	print(Ein, Eout)
	set_graph() # set graph
	
	plt.show()

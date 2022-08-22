import numpy as np
import matplotlib.pyplot as plt
import random

def set_graph(): # set display graph function
	plt.ylim(-1, 1)
	plt.xlim(-1, 1)
	plt.grid()
	plt.legend(["real function",'average'], loc='best')

def f(x): # input function
	return x**2

def real_func(start, stop, N): # real function
	x_real = np.linspace(start,stop, num=N)
	y_real = f(x_real)
	plt.plot(x_real,y_real)

def slope(x, y):
	slope_avg = ( ((y[1])/(x[1]))+((y[0])/(x[0])) )/2
	return slope_avg

def g(slope, x):
	return slope*x

def g_random(start, stop): # randomly data function
	x = np.array([random.uniform(start, stop), random.uniform(start, stop)])
	y = f(x)
	return x,y

def bias2(g_bar, fun):
	_bias2 = (g_bar-fun)**2
	return sum(_bias2)/len(_bias2)

def make_model(start, stop, N): # make avg. model
	sum_slope = 0
	slops = np.array([])
	x_vals = np.linspace(start,stop, num=N)
	
	for i in range(N):
		x, y = g_random(start, stop)
		
		_slope = slope(x,y)
		slops = np.append(slops, [_slope])
		
		sum_slope += _slope
	
	slope_avg = sum_slope/N
	
	g_avg = g(slope_avg, x_vals)
	
	bias2_avg = bias2(g_avg, f(x_vals))
	var = np.average( (g_avg-g(slops,x_vals))**2 )
	print(f"bias2: {bias2_avg}")
	print(f"var: {var}")
	print(f"var+bias2_avg: {var+bias2_avg}")
	
	plt.plot(x_vals, g_avg, '--')
	Ein = sum(  ( g(slops,x_vals)-f(x_vals) )**2  )/(2*N)
	Eout = var+bias2_avg
	return Ein, Eout
	
if __name__ == '__main__':
	N = 10000 # จำนวนรอบ
	start = -1 # x น้อยสุด
	stop = 1 # x มากสุด
	real_func(start, stop, N)
	Ein, Eout = make_model(start, stop, N)
	print(Ein, Eout)
	set_graph() # set graph
	
	plt.show()

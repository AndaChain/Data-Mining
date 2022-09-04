import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from random import uniform
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
random.seed(1)

def data(name):
	data = pd.read_csv(name)
	x = data.iloc[:, 0].to_numpy()
	y = data.iloc[:, 1].to_numpy()
	return x,y

def random_input(N, start, stop):# random input function
	x = np.array([])
	x = np.linspace(start, stop, N)
	
	f = lambda x: np.sin(np.pi*x) #10*random.uniform(0.5, stop)*x+10*random.uniform(start, stop)
	y = f(x)
	for n in range(N): # N is show number of sample
		y[n] = y[n]+random.uniform(0, 0.5)
	return x,y

class model:
	def __init__( self, learning_rate=0.01, iterations=1000, penality=1 ):
		self.learning_rate = learning_rate
		self.iterations = iterations        
		self.penality = penality
        
	def training(self, X, Y):
		# no_of_training_examples, no_of_features        
		self.m, self.n = X.shape
		
        # weight initialization        
		self.W = np.array([-0.5 for i in range(self.n)])
		self.W_hist = [self.W]
		self.b = 0 
		self.b_hist = [self.b]
		self.X = X
		self.Y = Y
		Y_pred = self.predict( self.X )
		self.rss_hist = [  - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) + ( 2 * self.penality * self.W )  ]
		
		# gradient descent learning 
		for i in range( self.iterations ) :            
			self.update_weights()
			self.W_hist.append(self.W)
			self.b_hist.append(self.b)
		
		self.W_hist = np.array(self.W_hist)
		self.b_hist = np.array(self.b_hist)
		self.rss_hist = np.array(self.rss_hist)
		
		return self

	def update_weights(self):
		Y_pred = self.predict( self.X )
		
		# calculate gradients
		# np.sum(((X.reshape(-1, 1).dot(self.W) + self.b) - Y) ** 2) + al * slope[i] ** 2
		rss = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) + ( 2 * self.penality * self.W ) )
		# print(self.W, self.b, rss)
		self.rss_hist.append(rss)
		dW = rss / self.m
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights
		self.W = self.W - self.learning_rate * dW
		self.b = self.b - self.learning_rate * db
		return self

	def predict(self, X):
		return X.dot( self.W ) + self.b

def x_y_al(tag):
	########Data#########
	x, y = random_input(1000, -1, 1)
	########Data#########

	arr_legend = np.array([])
	
	Alpha = np.arange(0, 1, 0.1)
	
	for alpha in Alpha:
		if(alpha == 0):
			########Linear#########
			Lge = LinearRegression()
			Lge.fit(x.reshape(-1, 1), y.reshape(-1, 1))

			w1 = Lge.coef_
			w0 = Lge.intercept_
			y_predict = Lge.predict(x.reshape(-1, 1))
			_x = x.reshape(-1, 1)
			plt.plot(_x, y_predict, "-")
			arr_legend = np.append(arr_legend, ["LinearReg"])
			continue
		if(tag == "ridge"):
			# R
			clf_R = Ridge(alpha=alpha)
			clf_R.fit(x.reshape(-1, 1), y.reshape(-1, 1))
			########
			
			# R
			w1_R = clf_R.coef_
			w0_R = clf_R.intercept_
			y_predict_R = clf_R.predict(x.reshape(-1, 1))
			_x_R = x.reshape(-1, 1)
			########

			plt.plot(_x_R, y_predict_R, "--")
			arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha)+", w1:"+str(np.round(w1_R[0][0], 3))])
		elif(tag == "lasso"):
			# L
			clf_L = Lasso(alpha=alpha)
			clf_L.fit(x.reshape(-1, 1), y.reshape(-1, 1))
			########
			
			# L
			w1_L = clf_L.coef_
			w0_L = clf_L.intercept_
			y_predict_L = clf_L.predict(x.reshape(-1, 1))
			_x_L = x.reshape(-1, 1)
			########
			
			plt.plot(_x_L, y_predict_L, "--")
			arr_legend = np.append(arr_legend, ["Alpha:"+str(alpha)+", w1:"+str(np.round(w1_L[0], 3))])
	
	plt.legend(arr_legend, loc='best')
	plt.xlabel('X axis')
	plt.ylabel('Y axis')
	plt.title('LinearRegression & '+tag)

	plt.show()

class model2:
	def __init__(self, tag="linear", al1=10, al2=10):
		self.n = 10000 # n is show number of model
		self.N = 2 # N is show number of sample
		self.posible_x = 2000
		self.mode = "non-constant"
		self.x_real = np.linspace(-1,1, num=self.posible_x)
		self.fun = lambda x : np.sin(x*np.pi)
		self.org = False
		self.tag = tag
		self.al1 = al1
		self.al2 = al2
	
	def f(self, x): # input function
		return self.fun(x)
	
	def set_n_model(self, n):
		self.n = n

	def set_N_sample(self, N):
		self.N = N
	
	def get_bias2(self): # show bias
		self._bias2 = (self.model_aveg-self.f(self.x_real))**2
		return sum(self._bias2)/len(self.x_real)

	def get_variance(self, models):# show variance
		sum_out = 0
		_index = 0
		for x in self.x_real:
			for g in models:
				_out = (g.predict(x.reshape(-1, 1))-self.model_aveg[_index])**2
				sum_out += _out
			_index += 1
		return sum_out/(len(self.x_real)*len(models))
	
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
		if(self.tag == "ridge"):
			return Ridge(alpha=self.al1).fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)
		elif(self.tag == "lasso"):
			return Lasso(alpha=self.al2).fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)
		return LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))#line_reg(x, y)

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
			x,y = self.random_input(self.N, -1, 1) # N is show number of model
			_model = self.create_model(x,y)
			#print( mean_squared_error(y, _model.predict(x.reshape(-1, 1))) )
			models = np.append(models, [_model])
			plt.plot( self.x_real, _model.predict(self.x_real.reshape(-1, 1)), "-", c='k' )
		return models

	def main(self):
		#arr_N = np.arange(2,21,1)
		#for N in arr_N:
		self.al1 = 0.5
		self.al2 = 0.5
		self.set_n_model(100) # number of model
		self.set_N_sample(2) # number of sample
		models = self.get_model()
		
		g_aveg = self.get_model_avg(models)
		
		if(g_aveg.shape[0] == 1):
			g_aveg = g_aveg[0]
		else:
			g_aveg = g_aveg
		self.model_aveg = g_aveg
		
		plt.plot( self.x_real, self.fun(self.x_real.reshape(-1, 1)), "-", linewidth=7.0)
		plt.plot( self.x_real, g_aveg, "-")
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		
		bias2 = self.get_bias2()
		variance = self.get_variance(models)
		################
		print(self.tag)
		if(self.tag == "lasso"):
			print("alpha:",self.al2)
		elif(self.tag == "ridge"):
			print("alpha:",self.al1)
		print("Bias: %.3f" % bias2)
		print("Variance: %.3f" % variance)
		print("E_out: %.3f" % (bias2 + variance))
		print("=========================")
		
		plt.show()
		
if __name__ == '__main__':
	X, Y = random_input(1000, -1,1)
	ridge = model(penality=0)
	ridge.training(X.reshape(-1, 1), Y)
	
	alpha = np.linspace(0, 900, 10)#np.arange(0, 1, 0.1)#[0, 20, 40, 60, 80]
	slope = ridge.W_hist
	intercept = ridge.b_hist
	x_line = np.array([])
	y_line = np.array([])
	
	for al in alpha:
		rss = np.array([np.sum(((X.reshape(-1, 1).dot(slope[i]) + intercept[i]) - Y) ** 2) + al * slope[i] ** 2 for i in range(slope.shape[0])])
		plt.plot(slope, rss, label=f"alpha={al}")
		index = np.where(rss == min(rss))
		plt.scatter(slope[index[0]], rss[index[0]])
		print(slope[index[0]], rss[index[0]])
		
		x_line = np.append(x_line, [slope[index[0]]])
		y_line = np.append(y_line, [rss[index[0]]])
		
		plt.legend()
	plt.xlabel("Slope Values")
	plt.ylabel("Sum of Squared Risidual + Penalty")
	plt.title("Ridge Regression")
	plt.show()
	
	for al in alpha:
		rss = np.array([np.sum(((X.reshape(-1, 1).dot(slope[i]) + intercept[i]) - Y) ** 2) + al * np.abs(slope[i]) for i in range(slope.shape[0])])
		plt.plot(slope, rss, label=f"alpha={al}")
		index = np.where(rss == min(rss))
		plt.scatter(slope[index[0]], rss[index[0]])
		
		x_line = np.append(x_line, [slope[index[0]]])
		y_line = np.append(y_line, [rss[index[0]]])
		
		plt.legend()
	plt.xlabel("Slope Values")
	plt.ylabel("Sum of Squared Risidual + Penalty")
	plt.title("Lasso Regression")
	plt.show()
	
	x_y_al("ridge")
	x_y_al("lasso")


	obj = model2()
	
	obj.tag = "linear"
	obj.main()

	
	obj.tag = "ridge"
	obj.main()
	
	obj.tag = "lasso"
	obj.main()

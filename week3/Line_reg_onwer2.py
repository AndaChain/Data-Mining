import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from random import uniform
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
random.seed(1)

def show(x,y,format_):
	#plt.plot(x, y, format_)
	#plt.show()
	plt.rcParams['figure.figsize'] = (12.0, 9.0)

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
		y[n] = y[n]+random.uniform(0, 0.001)
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
		rss = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) + ( 2 * self.penality * self.W ) )
		self.rss_hist.append(rss)
		dW = rss / self.m
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights
		self.W = self.W - self.learning_rate * dW
		self.b = self.b - self.learning_rate * db
		return self

	def predict(self, X):
		return X.dot( self.W ) + self.b
	
if __name__ == '__main__':
	X, Y = random_input(1000, -1,1)
	ridge = model(penality=0)
	ridge.training(X.reshape(-1, 1), Y)

	alpha = np.linspace(0, 900, 10)#[0, 20, 40, 60, 80]
	slope = ridge.W_hist
	intercept = ridge.b_hist
	x_line = np.array([])
	y_line = np.array([])
	
	for al in alpha:
		rss = np.array([np.sum(((X.reshape(-1, 1).dot(slope[i]) + intercept[i]) - Y) ** 2) + al * slope[i] ** 2 for i in range(slope.shape[0])])
		plt.plot(slope, rss, label=f"alpha={al}")
		index = np.where(rss == min(rss))
		plt.scatter(slope[index[0]], rss[index[0]])
		x_line = np.append(x_line, [slope[index[0]]])
		y_line = np.append(y_line, [rss[index[0]]])
		plt.legend()
	plt.xlabel("Slope Values")
	plt.ylabel("Sum of Squared Risidual + L2 Penalty")
	plt.title("Ridge Regression")
	plt.show()
	
	plt.plot(	x_line, y_line)
	plt.show()
	

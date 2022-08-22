import numpy as np
import matplotlib.pyplot as plt
import random
#from sklearn.linear_model import LinearRegression

class liner:
	def __init__(self, slope, intercept):
		self.slope = slope
		self.intercept = intercept
	
	def y(self, x):
		return self.intercept + (self.slope*x)

class constant:
	def __init__(self, x, f):
		self.y = sum(f(x))/len(f(x))

class model:
	def __init__(self):
		self.n = 10000 # n is show number of model
		self.N = 2 # N is show number of sample
		self.posible_x = 2000
		self.mode = "non-constant"
		self.x_real = np.linspace(-1,1, num=self.posible_x)
		self.fun = lambda x : np.sin(x*np.pi)
		self.org = False
	
	def f(self, x): # input function
		return self.fun(x)
	
	def set_n_model(self, n):
		self.n = n

	def set_N_sample(self, N):
		self.N = N

	def set_posible_x(self, posible_x):
		self.posible_x = posible_x
		self.x_real = np.linspace(-1,1, num=self.posible_x)

	def get_bias2(self, model_avg): # show bias
		if(self.mode == "non-constant"):
			self._bias2 = (model_avg.y(self.x_real)-self.f(self.x_real))**2
			return sum(self._bias2)/len(self.x_real)
		elif(self.mode == "constant"):
			self._bias2 = (model_avg-self.f(self.x_real))**2
			return sum(self._bias2)/len(self.x_real)

	def get_variance(self, models, model_avg):# show variance
		sum_out = 0
		sum_out2 = 0
		self.var_models = np.array([])
		if(self.mode == "non-constant"):
			for x in self.x_real:
				sum_out2 = 0
				for g in models:
					_out = (g.y(x)-model_avg.y(x))**2
					sum_out += _out
					sum_out2 += _out
				self.var_models = np.append(self.var_models, [sum_out2/len(models)])
			return sum_out/(len(self.x_real)*len(models))
		elif(self.mode == "constant"):
			for x in self.x_real:
				sum_out2 = 0
				for g in models:
					_out = (g.y-model_avg)**2
					sum_out += _out
					sum_out2 += _out
				self.var_models = np.append(self.var_models, [sum_out2/len(models)])
			return sum_out/(len(self.x_real)*len(models))

	def get_Eout(self, bias2, variance):
		Eout = bias2+variance
		return Eout

	def get_Ein(self, x, model):
		Ein = 0
		if(self.mode == "non-constant"):
			for i in x:
				Ein += ( model.y(i)-self.f(i) )**2
			return Ein
		elif(self.mode == "constant"):
			for i in x:
				Ein += ( model.y-self.f(i) )**2
			return Ein

	def get_model(self): # show bias
		models = np.array([])
		self.Ein = 0
		for i in range(self.n): # n is show number of model
			x,y = self.random_input(self.N, -1, 1)
			_model = self.create_model(x,y)
			models = np.append(models, [_model])
			self.Ein += self.get_Ein(x, _model)
		self.Ein = self.Ein/(self.n*self.N)
		return models

	def get_model_avg(self, models): # show model average
		sum_slope = 0
		sum_intercept = 0
		slope_avg = 0
		intercept_avg = 0
		model_avg = None
		
		if(self.mode == "non-constant"):
			for g in models:
				sum_slope += g.slope
				sum_intercept += g.intercept
			slope_avg = sum_slope/len(models)
			intercept_avg = sum_intercept/len(models)
			model_avg = liner(slope_avg, intercept_avg)
			return model_avg
		elif(self.mode == "constant"):
			out_sum = 0
			for i in range(len(self.x_real)):
				for j in range(len(models)):
					out_sum += self.f(self.x_real[i])-models[j].y
			return out_sum/(len(models)*len(self.x_real))

	def cal_slope_intercept(self, x, y):
		if(self.org):
			A = np.vstack([x, np.zeros(len(x))]).T # org
		else:
			A = np.vstack([x, np.ones(len(x))]).T # nornal
		slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
		return slope,intercept

	def random_input(self, N, start, stop):# random input function
		x = np.array([])
		for n in range(N): # N is show number of sample
			x = np.append(x, [random.uniform(start, stop)])
		y = self.f(x)
		return x,y
	
	def create_model(self, x, y):# show model
		if(self.mode == "non-constant"):
			_slope,_intercept = self.cal_slope_intercept(x,y)
			_model = liner(_slope,_intercept)
			return _model
		elif(self.mode == "constant"):
			_model = constant(x, self.f)
			return _model

	def main(self):
		sin = lambda x : np.sin(np.pi*x) 
		x_2 = lambda x : x**2
		fun = {"sin":sin, "x**2":x_2}
		
		org = [False, True]
		
		mode = ["non-constant","constant"]
		
		self.set_n_model(1000)
		arr_N = np.arange(2,21,1)
		for f in fun:
			self.fun = fun[f]
			for o in org:
				self.org = o
				for m in mode:
					arr_Eout = np.array([])
					arr_Ein = np.array([])
					self.mode = m
					print(f,o,m)
					for N in arr_N:
						self.set_N_sample(N)
							
						models = self.get_model()
						model_avg = self.get_model_avg(models)
						bias2 = self.get_bias2(model_avg)
						variance = self.get_variance(models, model_avg)
						Eout = self.get_Eout(bias2, variance)
						Ein = self.Ein

						arr_Eout = np.append(arr_Eout, [Eout])
						arr_Ein = np.append(arr_Ein, [Ein])

						print(f"N_sample:{self.N}")
						print(f"bias:{bias2},  variance:{variance}")
						print(f"Ein:{Ein}")
						print(f"Eout:{Eout}")
							
						
						if(N == 2 or N == 20):
							plt.plot(  self.x_real, self.f(self.x_real)  ) # plot real function
							plt.plot(  self.x_real, self._bias2  ) # plot bias2
							plt.plot(  self.x_real, self.var_models  ) # plot variance
							try:
								plt.plot(  self.x_real, model_avg.y(self.x_real)  ) # plot average
							except:
								plt.plot(  self.x_real, [model_avg]*len(self.x_real)  ) # plot average
							plt.grid()
							plt.legend(['real function','bias2','variance','average'], loc='best')
								
							plt.show()
							
							plt.cla()
						
						
					plt.plot(  arr_N, arr_Eout  )
					plt.plot(  arr_N, arr_Ein  )
					plt.grid()
					plt.legend(['Eout','Ein'], loc='best')
					plt.show()

		
		return Eout,Ein

if __name__ == '__main__':
	obj = model()
	
	obj.main()

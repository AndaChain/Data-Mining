import numpy as np
import matplotlib.pyplot as plt

arr_legend = []

def p(r, mu, ker):
	return np.exp(-0.5*((r-mu)/ker)**2)/(np.sqrt(2*np.pi*ker))

def f1():
	ker = 15 #std
	mu = 30  #mean

	start = -100
	end = 100

	r = np.linspace(start, end, num=end-start)
	p_ = p(r, mu, ker)

	max_p = p(r[0], mu, ker)
	max_r = r[0]
	for i in r:
		if(max_p < p(i, mu, ker)):
			max_p = p(i, mu, ker)
			max_r = i

	#plt.plot(max_r, max_p, "o")
	plt.annotate( "("+str(np.round(max_r, 3))+", "+str(np.round(max_p, 3))+")",(max_r,max_p) )
	plt.plot(r, p_)
	arr_legend.append("std: "+str(ker)+", "+ "mean: "+str(mu))

def f2():
	ker = 10 #std
	mu = 5 #mean

	start = -100
	end = 100

	r = np.linspace(start, end, num=end-start)
	p_ = p(r, mu, ker)

	max_p = p(r[0], mu, ker)
	max_r = r[0]
	for i in r:
		if(max_p < p(i, mu, ker)):
			max_p = p(i, mu, ker)
			max_r = i

	#plt.plot(max_r, max_p, "o")
	plt.annotate( "("+str(np.round(max_r, 3))+", "+str(np.round(max_p, 3))+")",(max_r,max_p) )
	plt.plot(r, p_)
	arr_legend.append("std: "+str(ker)+", "+ "mean: "+str(mu))

f1()
f2()
plt.legend(arr_legend, loc='best')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

start = -100
end = 100
ker1, mu1 = 5, 5
ker2, mu2 = 15, 30
Prior_1 = 0.5
Prior_2 = 1-Prior_1
r = np.linspace(start, end, num=end-start)

####### No.1->3 #######
def p(r, mu, ker):
	return np.exp(-0.5*((r-mu)/ker)**2)/(np.sqrt(2*np.pi*ker))

def f(ker, mu): # !!!No used!!!
	## ker = std
	## mu = mean

	#r = np.linspace(start, end, num=end-start)
	p_ = p(r, mu, ker)

	"""
	max_p = p(r[0], mu, ker)
	max_r = r[0]
	for i in r:
		if(max_p < p(i, mu, ker)):
			max_p = p(i, mu, ker)
			max_r = i
	"""

	#plt.plot(max_r, max_p, "o")
	#plt.annotate( "("+str(np.round(max_r, 3))+", "+str(np.round(max_p, 3))+")",(max_r,max_p) )
	plt.plot(r, p_)
	arr_legend.append("std: "+str(ker)+", "+ "mean: "+str(mu))
	return p_

def Evidence(r, arr_mu, arr_ker, arr_Prior):
	index = len(arr_mu)
	sum_of = 0
	for i in range(index):
		sum_of += p(r, arr_mu[i], arr_ker[i])*arr_Prior[i]
	return sum_of

def posteriori(r, mu, ker, Prior, Evi):
	Likelihood_i = p(r, mu, ker)
	posteriori_i = (Likelihood_i*Prior)/Evi # class i
	return posteriori_i

def g(r, mu, ker, Prior, Evi):
	return -((r - mu)**2)/(2*ker**2) - (1/2)*np.log(2*np.pi) - np.log(ker) + np.log(Prior) - np.log(Evi)

# Likelihood #################################################################
arr_legend = []
p_1 = p(r, ker1, mu1) # class 1
p_2 = p(r, ker2, mu2) # class 2
plt.plot(r, p_1)
plt.plot(r, p_2)
arr_legend.append("std: "+str(ker1)+", "+ "mean: "+str(mu1))
arr_legend.append("std: "+str(ker2)+", "+ "mean: "+str(mu2))

plt.legend(arr_legend, loc='best')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.title("Likelihood Single Variable, Basic Bayes")
plt.show() # Likelihood



# Posteriori #################################################################
# posteriori_i = (Likelihood_i*Prior_i)/(Evidence)
## Likelihood_i = p(r_i, mu_i, ker_i)
## Prior_i = constant_possible_val_i
## Evidence = sumofclass_i(Likelihood_i*Prior_i)

arr_legend = []

arr_mu = [mu1, mu2]
arr_ker = [ker1, ker2]
arr_Prior = [Prior_1, Prior_2]

#r = np.linspace(start, end, num=end-start)
Evi = Evidence(r, arr_mu, arr_ker, arr_Prior)
posteriori_1 = posteriori(r, mu1, ker1, Prior_1, Evi) # class 1
posteriori_2 = posteriori(r, mu2, ker2, Prior_2, Evi) # class 2
plt.plot(r, posteriori_1)
plt.plot(r, posteriori_2)
arr_legend.append("std: "+str(ker1)+", "+ "mean: "+str(mu1))
arr_legend.append("std: "+str(ker2)+", "+ "mean: "+str(mu2))

plt.legend(arr_legend, loc='best')
plt.xlabel('x')
plt.ylabel('Posteriori')
plt.title("Posteriori Single Variable, Basic Bayes")
plt.show() # Posteriori



# Decision Boudary, Basic Bayes#################################################################
#r = np.linspace(start, end, num=end-start)
Boudary = np.log(posteriori_1) - np.log(posteriori_2)#np.log( p(r, mu1, ker1)/p(r, mu2, ker2) ) + np.log( Prior_1/Prior_2 )
print(Boudary)
plt.plot(r, Boudary)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Decision Boudary Single Variable, Basic Bayes")
plt.show()



# Decision Boudary, Quadratic Function#################################################################
g1 = g(r, mu1, ker1, Prior_1, Evi)
g2 = g(r, mu2, ker2, Prior_2, Evi)
Boudary_Qu = g1 - g2
print(Boudary_Qu)
plt.plot(r, Boudary_Qu)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Decision Boudary Single Variable, Quadratic Function")
plt.show()


figure, axis = plt.subplots(1, 2)
axis[0].plot(r, Boudary)
axis[0].set_title("Basic Bayes")
  
axis[1].plot(r, Boudary_Qu)
axis[1].set_title("Quadratic Function")
plt.show()

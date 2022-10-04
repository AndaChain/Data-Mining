import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(1)

# parameter
start = -10
end = 10
ker1, mu1 = 2, -30
ker2, mu2 = 2, 30
Prior_1 = 0.1
Prior_2 = 1-Prior_1
r = np.linspace(start, end, num=1000)

############### No.4 ###############

def holdout(x,test_size):
	#x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y.reshape(-1, 1), test_size=test_size, random_state=seed)
	N_test_size = len(x)*test_size
	print(N_test_size)
	_x = np.random.choice(range(len(x)), size=int(N_test_size), replace=False)
	print(_x)
	class1 = []
	class2 = []
	
	for i in range(len(x)):
		if(i in _x):
			class1.append(x[i])
		else:
			class2.append(x[i])
	
	class1 = np.array(class1)
	class2 = np.array(class2)
	return class1, class2
	
if(input()=="1"):
	ran = r#np.linspace(start, end, num=1000)
	class_1, class_2 = holdout(ran,test_size=0.5)
	print(len(class_1), len(class_2))
	ker1, mu1 = class_1.std(), class_1.mean()
	ker2, mu2 = class_2.std(), class_2.mean()
	Prior_1 = len(class_1)/len(ran)
	Prior_2 = len(class_2)/len(ran)
	plt.plot(class_1, len(class_1)*[0], "*", color = 'red')
	plt.plot(class_2, len(class_2)*[1], "o", color = 'blue')
	print("std1: "+str(ker1)+", "+ "mean1: "+str(mu1))
	print("std2: "+str(ker2)+", "+ "mean2: "+str(mu2))
	print("Prior1: "+str(Prior_1)+", "+ "Prior2: "+str(Prior_2))
	plt.xlabel('x')
	plt.ylabel('Classes')
	plt.yticks([0,1])
	
	
	plt.show()
##################################

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

####################

# Likelihood #################################################################
arr_legend = []
p_1 = p(r, mu1, ker1) # class 1
p_2 = p(r, mu2, ker2) # class 2
plt.plot(r, p_1)
plt.plot(r, p_2)
arr_legend.append("std: "+str(ker1)+", "+ "mean: "+str(mu1))
arr_legend.append("std: "+str(ker2)+", "+ "mean: "+str(mu2))

plt.legend(arr_legend, loc='best')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.title("Likelihood Single Variable")
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
plt.title("Posteriori Single Variable")
plt.show() # Posteriori



# Decision Boudary, Basic Bayes#################################################################
#r = np.linspace(start, end, num=end-start)
Boudary = []
if(Prior_1 >= Prior_2):
	Boudary = np.log(posteriori_1) - np.log(posteriori_2)#np.log( p(r, mu1, ker1)/p(r, mu2, ker2) ) + np.log( Prior_1/Prior_2 )
else:
	Boudary = np.log(posteriori_2) - np.log(posteriori_1)
print(Boudary)
plt.plot(r, Boudary)
plt.xlabel('x')
plt.ylabel('y')
try:
	plt.plot(class_1, len(class_1)*[0], "*", color = 'red')
	plt.plot(class_2, len(class_2)*[1], "o", color = 'blue')
except:
	pass
plt.title("Decision Boudary Single Variable")
plt.show()



# Decision Boudary, Quadratic Function#################################################################
g1 = g(r, mu1, ker1, Prior_1, Evi)
g2 = g(r, mu2, ker2, Prior_2, Evi)
Boudary_Qu = []
if(Prior_1 >= Prior_2):
	Boudary_Qu = g1 - g2
else:
	Boudary_Qu = g2 - g1
print(Boudary_Qu)
plt.plot(r, Boudary_Qu)
plt.xlabel('x')
plt.ylabel('y')
try:
	plt.plot(class_1, len(class_1)*[0], "*", color = 'red')
	plt.plot(class_2, len(class_2)*[1], "o", color = 'blue')
except:
	pass
plt.title("Decision Boudary Single Variable, Quadratic Function")
plt.show()



figure, axis = plt.subplots(1, 2)
plt.title("Number of Sample: 10")
axis[0].plot(r, Boudary)
axis[0].set_title("Linear Function")
try:
	axis[0].plot(class_1, len(class_1)*[0], "*", color = 'red')
	axis[0].plot(class_2, len(class_2)*[1], "o", color = 'blue')
except:
	pass

axis[1].plot(r, Boudary_Qu)
axis[1].set_title("Quadratic Function")
try:
	axis[1].plot(class_1, len(class_1)*[0], "*", color = 'red')
	axis[1].plot(class_2, len(class_2)*[1], "o", color = 'blue')
except:
	pass
plt.show()


# Mybe Special #################################################







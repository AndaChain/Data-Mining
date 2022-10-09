import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(1)

########## genarate data random&from mean AND std ##########
def genarate_data(desired_mean, desired_std_dev, num_samples):
	desired_mean = desired_mean
	desired_std_dev =desired_std_dev
	num_samples = num_samples

	samples = np.random.normal(size=num_samples)
	#samples = np.array(range(0,num_samples))
	actual_mean = np.mean(samples)
	actual_std = np.std(samples)
	
	zero_mean_samples = samples - (actual_mean)
	zero_mean_std = np.std(zero_mean_samples)
	scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std) ###### key
	
	final_samples = scaled_samples + desired_mean
	final_mean = np.mean(final_samples)
	final_std = np.std(final_samples)
	
	return final_samples

# parameter & data
start = -100
end = 100
N = end-start
mu1, ker1 = -30, 15
mu2, ker2 = 30, 15
Prior_1 = 0.5
Prior_2 = 1-Prior_1
r = np.linspace(start, end, num=end-start)
class_1 = genarate_data(mu1, ker1, int(N*Prior_1))
class_2 = genarate_data(mu2, ker2, N-int(N*Prior_1))

#plt.plot(class_1, len(class_1)*[0], "o")
#plt.plot(class_2, len(class_2)*[1], "o")
#plt.show()

############### No.4 ###############
def holdout(x,test_size):
	N_test_size = len(x)*test_size
	_x = np.random.choice(range(len(x)), size=int(N_test_size), replace=False)
	return x[_x]
	
if(input()=="1"):
	want = 100
	ratio = want/N
	class_1 = holdout(class_1, ratio)
	class_2 = holdout(class_2, ratio)
	
	
	plt.plot(class_1, len(class_1)*[0], "o")
	plt.plot(class_2, len(class_2)*[1], "o")

	mu1, ker1 = class_1.mean(), class_1.std()
	mu2, ker2 = class_2.mean(), class_2.std()
	
	plt.legend(["class_1","class_2"], loc='best')
	plt.xlabel('x')
	plt.ylabel('classes')
	plt.yticks([0,1])
	plt.title(str(want)+" Data")
	plt.show()
##################################

####### No.1->3 #######
def p(r, mu, ker):
	return np.exp(-0.5*((r-mu)/ker)**2)/(np.sqrt(2*np.pi*ker))

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
p_1 = p(r, mu1, ker1) # class 1
p_2 = p(r, mu2, ker2) # class 2
plt.plot(r, p_1)
plt.plot(r, p_2)
arr_legend.append("std: "+str(np.round(ker1,2))+", "+ "mean: "+str(np.round(mu1,2)))
arr_legend.append("std: "+str(np.round(ker2,2))+", "+ "mean: "+str(np.round(mu2,2)))

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
arr_legend.append("std: "+str(np.round(ker1,2))+", "+ "mean: "+str(np.round(mu1,2)))
arr_legend.append("std: "+str(np.round(ker2,2))+", "+ "mean: "+str(np.round(mu2,2)))

plt.legend(arr_legend, loc='best')
plt.xlabel('x')
plt.ylabel('Posteriori')
plt.title("Posteriori Single Variable")
plt.show() # Posteriori

"""
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
"""







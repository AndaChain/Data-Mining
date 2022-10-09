import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(1)

class LDA:
	def fit(self, X, t):
		self.priors = dict()
		self.means = dict()
		self.cov = np.cov(X, rowvar=False)
        
		self.classes = np.unique(t)

		for c in self.classes:
			X_c = X[t == c]
			self.priors[c] = X_c.shape[0] / X.shape[0]
			self.means[c] = np.mean(X_c, axis=0)
            
	def predict(self, X):
		preds = list()
		arr_likelihoods = [] ########
		arr_posts = [] ########
        
		for x in X:
			posts = list()
			arr_likelihood = [] ########
			for c in self.classes:
				prior = np.log(self.priors[c])
				inv_cov = np.linalg.inv(self.cov)
				inv_cov_det = np.linalg.det(inv_cov)
				diff = x-self.means[c]
				likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
				arr_likelihood.append(likelihood) ########
				post = prior + likelihood
				posts.append(post)
			arr_likelihoods.append(arr_likelihood) ########
			arr_posts.append(posts) ########
			pred = self.classes[np.argmax(posts)]
			preds.append(pred)
		arr_likelihoods = np.array(arr_likelihoods)
		arr_posts = np.array(arr_posts)
		
		return np.array(preds), arr_likelihoods, arr_posts

class QDA:
	def fit(self, X, t):
		self.priors = dict()
		self.means = dict()
		self.covs = dict()
		self.classes = np.unique(t)
		for c in self.classes:
			X_c = X[t == c]
			self.priors[c] = X_c.shape[0] / X.shape[0]
			self.means[c] = np.mean(X_c, axis=0)
			self.covs[c] = np.cov(X_c, rowvar=False)
		
	def predict(self, X):
		preds = list()
		arr_likelihoods = [] ########
		arr_posts = [] ########
		
		for x in X:
			#print(x)
			posts = list()
			arr_likelihood = [] ########
			for c in self.classes:
				prior = np.log(self.priors[c])
				inv_cov = np.linalg.inv(self.covs[c])
				inv_cov_det = np.linalg.det(inv_cov)
				diff = x-self.means[c]
				likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
				arr_likelihood.append(likelihood) ########
				post = prior + likelihood
				posts.append(post)
			arr_likelihoods.append(arr_likelihood) ########
			arr_posts.append(posts) ########
			pred = self.classes[np.argmax(posts)]
			preds.append(pred)
		arr_likelihoods = np.array(arr_likelihoods)
		arr_posts = np.array(arr_posts)
		
		return np.array(preds), arr_likelihoods, arr_posts

"""
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target
"""

data = pandas.read_csv("GenderHeightWeight.csv")

X = data.iloc[:, 1:3].values
y = data.iloc[:,[0]].values.reshape(1,-1)[0]


N_test_size = len(X)*1
index = np.random.choice(range(len(X)), size=int(N_test_size), replace=False)
X = X[index]
y = y[index]

bound = LDA()
bound.fit(X, y)

arr_likelihoods = bound.predict(X)[1]
arr_posts = bound.predict(X)[2]
ax = plt.axes(projection='3d')
ax.scatter(X[:,0], X[:,1], arr_likelihoods[:,0]) ########### class Female
ax.scatter(X[:,0], X[:,1], arr_likelihoods[:,1]) ########### class male
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Likelihoods')
plt.show()
		
ax = plt.axes(projection='3d')
ax.scatter(X[:,0], X[:,1], arr_posts[:,0]) ########### class Female
ax.scatter(X[:,0], X[:,1], arr_posts[:,1]) ########### class male
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Posteriori')
plt.show()

#################plot decision boundary#####################
#######1
h = 0.1  # step size in the mesh

x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# create a mesh to plot in
x0, x1= np.meshgrid(  np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h)  )
print(1)
######2
Z = bound.predict(np.c_[x0.ravel(), x1.ravel()])[0]
print(2)

######3
copy_Z = Z.copy()
for i in range(len(Z)):
	if(Z[i] == "Male"):
		copy_Z[i] = 1
	else:
		copy_Z[i] = 0

copy_y = y.copy()
for i in range(len(y)):
	if(y[i] == "Male"):
		copy_y[i] = 1
	else:
		copy_y[i] = 0

copy_Z = copy_Z.reshape(x0.shape)
plt.contourf(x0, x1, copy_Z, cmap=plt.cm.coolwarm, alpha=0.8)
print(3)
######4
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=copy_y, cmap=plt.cm.coolwarm)
plt.xlabel('X0')
plt.ylabel('X1')
plt.savefig("decision boundary")
plt.show()
print(4)

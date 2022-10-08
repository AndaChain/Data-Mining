import numpy as np
import pandas
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
		for x in X:
			posts = list()
			for c in self.classes:
				prior = np.log(self.priors[c])
				inv_cov = np.linalg.inv(self.covs[c])
				inv_cov_det = np.linalg.det(inv_cov)
				diff = x-self.means[c]
				likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
				post = prior + likelihood
				posts.append(post)
			pred = self.classes[np.argmax(posts)]
			preds.append(pred)
		return np.array(preds)


data = pandas.read_csv("GenderHeightWeight.csv")

X = data.iloc[:, 1:3].values
y = data.iloc[:,[0]].values.reshape(1,-1)[0]
qda = QDA()
qda.fit(X, y)
preds = qda.predict(X)
print(preds)
"""
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

X = data[:, 0:2]
t = data[:, 2]

print(t)

qda = QDA()
qda.fit(X, t)
preds = qda.predict(X)
"""

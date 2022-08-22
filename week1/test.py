import numpy as np

x1 = np.array([5,1,8,2,9])
x2 = x1.copy()
x2.resize(len(x2),1)
N = len(x1)

one = np.ones((N,), dtype=int)
one.resize(len(one),1)

a = np.multiply(x1,x2)
b = np.multiply(a,one)
print(b)
c = b.dot(one)
c.resize(1,len(c))
result = c.dot(one)[0][0]
print(result)

aa = 0
for i in x1:
	for j in x2:
		aa += i*j[0]
print(aa)


import matplotlib.pylab as plt
import numpy as np
# линейная регрессия

t = np.array([[1,2], [4,3], [2,4], [7,9], [7,10]])

plt.scatter(t[:,1], t[:,0])
y = t[:, 0]
x = t[:, 1]
X = np.copy(t)
X[:, 0] = np.ones(len(X[:, 0]))  # [1(w0), x1(w1)]!!!
step1 = X.T.dot(X)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X.T)
b = step3.dot(y)
b0 = 0
yj = b[0] + b[1] * x + b0  # [1(w1), x1(w1)]!!!
plt.plot(x, yj)
plt.grid(True)
print(X)
print(y)
print(b)
print(yj)

plt.show()

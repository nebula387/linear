import matplotlib.pylab as plt
import numpy as np
# линейная регрессия
# t[x, y]
t = np.array([[2,1], [3,4], [4,2], [5,7], [6,2], [7,4], [8,9], [9,7], [10,7]])

plt.scatter(t[:,0], t[:,1])
y = t[:, 1]
x = t[:, 0]
X = np.copy(t)
X[:, 1] = np.ones(len(X[:, 1]))  # [x(w1), 1(w0)]!!!
step1 = X.T.dot(X)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X.T)
b = step3.dot(y)
b0 = 0
yj = b[1] + b[0] * x + b0  # [x(w1), 1(w0)]!!!
plt.plot(x, yj)
plt.grid(True)
print(X)
print(y)
print(b)
print(yj)

plt.show()

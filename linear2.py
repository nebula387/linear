import matplotlib.pylab as plt
import numpy as np


def linear(t, b0=0):
    """ Принимает матрицу ([[y1, x1], [yn, xn]]) и возвращает
        график линейной регрессии и значение y при x=100 в %
        b0 = 0 по умолчанию"""
    points = plt.scatter(t[:, 1], t[:, 0])  # points

    y = t[:, 0]
    x = t[:, 1]
    X = np.copy(t)
    X[:, 0] = np.ones(len(X[:, 0]))  # array[1(w0), x1(w1)]!!!
    s1 = X.T.dot(X)
    s2 = np.linalg.inv(s1)
    s3 = s2.dot(X.T)
    b = s3.dot(y)  # [w0, w1]
    yj = b[0] * 1 + b[1] * x + b0  # predicted y!

    line = plt.plot(x, yj,  label=r'$b(,y) = (X^TX)^-1 X^Ty$')  # linear
    points.set(facecolor='#f26')
    # points.set(facecolor='#AAFFAA')
    plt.legend()
    plt.grid(True)

    xr = 100
    yr = b[0] + b[1] * xr + b0
    plt.text(min(x), max(y)-0.5, str(round(yr, 2))+'%')
    plt.show()


if __name__ == "__main__":

    t1 = np.array([[7, 10], [2, 3], [3, 5], [2, 3], [8, 10], [9, 10], [12, 14], [20, 20]])
    linear(t1)

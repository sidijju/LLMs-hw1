import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def part2():
    noise = torch.normal(0, 0.3, size=(1,20))[0]
    x = 10 * torch.rand((1, 20))
    x, _ = torch.sort(x)
    y = x + torch.sin(1.5 * x) + noise
    x_smooth = torch.linspace(0, 10, 100)
    fx = x_smooth + torch.sin(1.5 * x_smooth)
    plt.scatter(x, y, label='y', color='r')
    # f_cubic = interp1d(x, y, kind='cubic')
    plt.plot(x_smooth.tolist(), fx.tolist(), label='f(x)')
    plt.legend()
    plt.savefig('q1-part2.png')
    return x.numpy()[0], y.numpy()[0], x_smooth.numpy(), fx.numpy()

def fit_polynomial_regression(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return X @ coeffs

def part3(x, y, x_smooth, fx):
    g1 = fit_polynomial_regression(x, y, 1)
    g3 = fit_polynomial_regression(x, y, 3)
    g10 = fit_polynomial_regression(x, y, 10)
    plt.cla()
    plt.plot(x_smooth, fx, label='f(x)')
    plt.plot(x, g1, label='g1')
    plt.plot(x, g3, label='g3')
    plt.plot(x, g10, label='g10')
    plt.scatter(x, y, label='y', color='r')
    plt.legend()
    plt.savefig('q1-part3.png')

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    x, y, x_smooth, fx = part2()
    part3(x, y, x_smooth, fx)
    
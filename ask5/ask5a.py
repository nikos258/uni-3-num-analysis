import math

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


def lagrange_polynomial(points):
    """
    Calculates the Lagrange polynomial based on a list of given points.
    :param points: the points on the cartesian plane
    :return:
    """
    n = len(points)
    lagrange = Polynomial(np.zeros(n))  # makes a zero polynomial of degree n-1

    for i in range(n):
        # calculates the i-th lagrange coefficient
        poly = Polynomial(np.identity(n)[0])
        for j in range(n):
            if i != j:
                k = Polynomial((-points[j][0], 1))  # polynomial(-x_j + x)
                m = (points[i][0]-points[j][0])  # x_i - x_j
                poly *= k / m

        lagrange += points[i][1] * poly  # adds the quantity y_i times the i-th lagrange coefficient to the final
        # lagrange polynomial

    return lagrange


points = ((-np.pi, 0.), (-2.5, -0.59847), (-np.pi/2, -1.), (-1.4, -0.98544), (-0.6, -0.56464), (0., 0.), (1., 0.84147), (np.pi/2, 1.), (2.7, 0.42737), (np.pi, 0.))
p = lagrange_polynomial(points)

t = np.linspace(-np.pi, np.pi, 200)

error = list()
for i in range(200):
    sin = round(np.sin(t[i], dtype=np.float64), 5)
    error.append(abs(p(t[i]) - sin))

print("Maximum absolut value of the error: {:.7f}".format(max(error), dtype=np.float64))
print("Maximum absolut value of the error: {:.7f}".format(np.mean(error, dtype=np.float64)))

plt.plot(t, error)
plt.title("Absolut value of the error")
plt.show()

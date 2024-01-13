import numpy as np


def trapezoid_method(function, a, b, n):
    """
    Calculates an approximation of the integral of a given function over the interval [a, b] using the trapezoid method.
    :param function: the function
    :param a: the starting point of the interval
    :param b: the ending point of the interval
    :param n: the number of even partitions of the interval
    :return: the approximate value of the integral
    """
    partition = np.linspace(a, b, n+1)  # makes an even partition of n+1 points in the interval [a, b]

    # calculates the sum of f(x_i) for i in [1, n-1]
    approximation = 0
    for i in range(1, n):
        approximation += function(partition[i])

    approximation = function(partition[0]) + function(partition[n]) + 2*approximation
    approximation *= (b-a)/(2*n)
    return approximation


print("The value of the integral: ", trapezoid_method(np.sin, 0, np.pi/2, 10))

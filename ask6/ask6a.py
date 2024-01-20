import numpy as np


def simpson_method(function, a, b, n):
    """
    Calculates an approximation of the integral of a given function over the interval [a, b] using Simpson's method.
    :param function: the function
    :param a: the starting point of the interval
    :param b: the ending point of the interval
    :param n: the number of even partitions of the interval (n must be an even number)
    :return: the approximate value of the integral
    """
    partition = np.linspace(a, b, n+1)

    s1 = 0
    for i in range(1, int(n/2)):
        s1 += function(partition[2*i])

    s2 = 0
    for i in range(1, int(n/2)+1):
        s2 += function(partition[2*i-1])

    return (b-a)/(3*n) * (function(partition[0]) + function(partition[n]) + 2*s1 + 4*s2)


print("The value of the integral: ", simpson_method(np.sin, 0, np.pi/2, 10))

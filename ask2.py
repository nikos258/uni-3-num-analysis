import numpy as np
import matplotlib.pyplot as plt
import random


def f(x): return 54*x**6+45*x**5-102*x**4-69*x**3+35*x**2+16*x-4


t = np.arange(-2., 2., 0.01)
plt.plot(t, f(t))
plt.grid()
# plt.show()


def newton_raphson2(f, df, d2f, x0, tol):
    if f(x0) == 0:
        return x0, 0

    steps = 1
    x_n = x0
    x_n1 = x_n - 1/(df(x_n)/f(x_n) - d2f(x_n)/(2*df(x_n)))
    while abs(x_n1-x_n) > tol and f(x_n1) != 0:
        x_n = x_n1
        x_n1 = x_n - 1/(df(x_n)/f(x_n) - d2f(x_n)/(2*df(x_n)))
        steps += 1
    return x_n1, steps


def bisection2(f, a, b, tol):
    steps = 0
    m = a
    while (b-a) > tol:
        steps += 1
        m = np.random.uniform(a, b)
        if f(m) == 0:
            return m, steps
        if np.sign(f(a)) * np.sign(f(m)) < 0:
            b = m
        else:
            a = m

    return m, steps


def secant2(f, x0, x1, x2, tol):
    steps = 1
    x_n = x0
    x_n1 = x1
    x_n2 = x2
    q = f(x_n) / f(x_n1)
    r = f(x_n2) / f(x_n1)
    s = f(x_n2) / f(x_n)
    x_n3 = x_n2 - (r*(r-q)*(x_n2-x_n1) + (1-r)*s*(x_n2-x_n)) / ((q-1)*(r-1)*(s-1))

    while abs(x_n3 - x_n2) > tol and f(x_n3) != 0:
        x_n = x_n1
        x_n1 = x_n2
        x_n2 = x_n3
        q = f(x_n) / f(x_n1)
        r = f(x_n2) / f(x_n1)
        s = f(x_n2) / f(x_n)
        x_n3 = x_n2 - (r*(r-q)*(x_n2-x_n1) + (1-r)*s*(x_n2-x_n)) / ((q-1)*(r-1)*(s-1))
        steps += 1

    return x_n3, steps


def df(x): return 324*x**5+225*x**4-408*x**3-207*x**2+70*x+16


def d2f(x): return 1620*x**4+900*x**3-1224*x**2-414*x+70


# Subsection 1

tolerance = 0.5*10**(-5)
print("Variation of the Newton-Raphson method:")
print(newton_raphson2(f, df, d2f, -2, tolerance))
print(newton_raphson2(f, df, d2f, -1, tolerance))
print(newton_raphson2(f, df, d2f, 0., tolerance))
print(newton_raphson2(f, df, d2f, 0.9, tolerance))
print(newton_raphson2(f, df, d2f, 2, tolerance))


print("\nVariation of the Bisection method:")
print(bisection2(f, -1.6, -1.2, tolerance))
print(bisection2(f, 0., 0.4, tolerance))
print(bisection2(f, 0.3, 0.7, tolerance))
print(bisection2(f, 1., 1.4, tolerance))

print("\nVariation of the Secant method:")
print(secant2(f, -1.6, -1.4, -1.2, tolerance))
print(secant2(f, -0.8, -0.6, -0.4, tolerance))
print(secant2(f, 0., 0.2, 0.4, tolerance))
print(secant2(f, 0.3, 0.4, 0.7, tolerance))
print(secant2(f, 1., 1.2, 1.4, tolerance))


# Subsection 2

iterations = set()
for i in range(20):
    iterations.add(bisection2(f, -1.6, -1.2, tolerance)[1])

# print("Different iterations of the variation of the bisection method:", iterations)


# Subsection 3

def bisection(f, a, b, tol):
    """
    Implements the method of bisection.
    :param f: the function
    :param a: the starting point of the interval
    :param b: the ending point of the interval
    :param tol: the tolerance
    :return: the calculated root and the actual number of steps the algorithm was executed
    """
    steps = 0
    m = a
    while (b-a)/2 > tol:
        steps += 1
        m = (a+b)/2
        if f(m) == 0:
            return m, steps
        if np.sign(f(a)) * np.sign(f(m)) < 0:
            b = m
        else:
            a = m

    return m, steps


def newton_raphson(f, df, x0, tol, root_multiplicity=1):
    """
    Implements the Newton-Raphson method or the variation of the method in case root_multiplicity>1
    :param f: the function
    :param df: the derivative of the function
    :param x0: the starting point
    :param tol: the tolerance
    :param root_multiplicity: the multiplictiy of the root to be calculated (default is 1)
    :return: the calculated root and the actual number of steps the algorithm was executed
    """
    if f(x0) == 0:
        return x0, 0

    steps = 1
    x_n = x0
    x_n1 = x_n - root_multiplicity*f(x_n)/df(x_n)
    while abs(x_n1-x_n) > tol and f(x_n1) != 0:
        x_n = x_n1
        x_n1 = x_n - root_multiplicity*f(x_n)/df(x_n)
        steps += 1
    return x_n1, steps


def secant(f, x0, x1, tol):
    """
    Implements the secant method
    :param f: the function
    :param x0: the first starting point
    :param x1: the second starting point
    :param tol: the tolerance
    :return: the calculated root and the actual number of steps the algorithm was executed
    """
    steps = 1
    x = x0
    x_n = x1
    x_n1 = x_n - f(x_n)*(x_n-x)/(f(x_n)-f(x))
    while abs(x_n1 - x_n) > tol and f(x_n1) != 0:
        x = x_n
        x_n = x_n1
        x_n1 = x_n - f(x_n)*(x_n-x)/(f(x_n)-f(x))
        steps += 1

    return x_n1, steps



print("\nNormal Newton-Raphson")
print(newton_raphson(f, df, -2, tolerance))
print(newton_raphson(f, df, -1, tolerance))
print(newton_raphson(f, df, 0., tolerance))
print(newton_raphson(f, df, 0.9, tolerance))
print(newton_raphson(f, df, 2, tolerance))

print("\nNormal Bisection Method")
print(bisection(f, -1.6, -1.2, tolerance))
print(bisection(f, 0., 0.4, tolerance))
print(bisection(f, 0.3, 0.7, tolerance))
print(bisection(f, 1., 1.4, tolerance))

print("\nNormal Secant Method")
print(secant(f, -1.6, -1.2, tolerance))
print(secant(f, -0.8, -0.4, tolerance))
print(secant(f, 0., 0.4, tolerance))
print(secant(f, 0.3, 0.7, tolerance))
print(secant(f, 1., 1.4, tolerance))

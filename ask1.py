import numpy as np
import matplotlib.pyplot as plt
import math

e = math.e


def f(x): return 14*x*e**(x-2)-12*e**(x-2)-7*x**3+20*x**2-26*x+12


t = np.arange(0., 3., 0.01)
plt.plot(t, f(t))
plt.grid()
# plt.show()


# a) Bisection method
def get_number_of_steps(digits_of_accuracy, a, b):
    """
    :param digits_of_accuracy: the number of decimal digits of accuracy
    :param a: the starting point of the interval
    :param b: the ending point of the interval
    :return: the number of steps needed to guarantee a specific number of digits of accuracy
    """
    return math.ceil(math.log2((b-a)*10**(digits_of_accuracy+1)/5))


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


tolerance = 0.5*10**(-5)  # tolerance of 5 decimal digits

print("Bisection method:")
print(bisection(f, 0., 1., tolerance))
print(bisection(f, 1., 3., tolerance))


# Newton-Raphson method


def df(x): return 2*e**(x-2)*(7*x+1)-21*x**2+40*x-26
def d2f(x): return 2*e**(x-2)*(7*x+8)-42*x+40
def d3f(x): return 2*e**(x-2)*(7*x+15)-42


plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, df(t))
ax1.grid()
ax1.set_title("f'(x)")
ax2.plot(t, d2f(t))
ax2.grid()
ax2.set_title("f''(x)")
# plt.show()


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


print('\nNewton-Raphson method:')
print(newton_raphson(f, df, 0., tolerance))
print(newton_raphson(f, df, 3., tolerance))
print(newton_raphson(f, df, 3., tolerance, 3))


# Secant method


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


print('\nSecant method:')
print(secant(f, 0., 1., tolerance))
print(secant(f, 1., 3., tolerance))
print(secant(f, 1.5, 2.5, tolerance))

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


def bisection(f, a, b, steps):
    """
    Implements the method of bisection.
    :param f: the function
    :param a: the starting point of the interval
    :param b: the ending point of the interval
    :param steps: the number of steps the algorithm was executed
    :return: the calculated root and the actual number of steps the algorithm was executed
    """
    for i in range(steps):
        m = (a+b)/2.0
        if f(m) == 0:
            return m, i+1
        if np.sign(f(a))*np.sign(f(m)) < 0:
            b = m
        else:
            a = m
    return m, i+1


steps = get_number_of_steps(5, 0., 3.)

print("Bisection method:")
print(bisection(f, 0., 1.1, steps))
print(bisection(f, 1.1, 3., steps))

# Newton-Raphson method



import numpy as np
from matplotlib import pyplot as plt


def power(A, tol):
    """
    Implements the power method to find the eigenvalue with the largest absolut value and the respective eigenvalue of
    the matrix A.
    :param A: the matrix
    :param tol: the toleration
    :return: the calculated eigenvector and eigenvalue
    """
    shape = A.shape[0]
    b = np.ones(shape)
    l = b[0]
    condition = tol + 1  # this condition ensures at least one iteration of the while loop

    while condition > tol:
        l_before = l
        b = np.dot(A, b)

        # finds the first non-zero value of the vector
        i = 0
        while i < shape:
            if b[i] == 0:
                i += 1
            else:
                l = b[i]
                break
        else:
            return b, 0

        b /= l  # normalizes the vector with the calculated eigenvalue
        condition = abs(l - l_before)

    return b, l_before


def get_google(A, q):
    n = A.shape[0]
    # calculates the sum of every row of the matrix A
    row_sum = np.zeros(n)
    for i in range(n):
        row_sum[i] = sum(A[i])

    # calculates the Google matrix G
    G = np.identity(n)
    for i in range(n):
        for j in range(n):
            G[i, j] = q/n + A[j, i] * (1-q) / row_sum[j]

    return G


# 4.2

toleration = 0.5 * 10**(-15)

A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]])

q = 0.15
G = get_google(A, q)
p, eigenvalue = power(G, toleration)
p = p/np.sum(p)

# print("4.2:")
# print("The normalised eigenvector: ", p)
# print("The eigenvalue: ", eigenvalue)


# 4.3

A3 = A.copy()
A3[10-1, 11-1] = 1
A3[11-1, 1-1] = 1
A3[11-1, 15-1] = 0
# A3[10, 9] = 1
# A3[11-1, 15-1] = 0

G3 = get_google(A3, q)

p3, eigenvalue3 = power(G3, toleration)
p3 = p/np.sum(p3)
# print(A3)
# print(p3, eigenvalue3)
# print(p3[1-1], p[1-1])


# 4.4

# q = 0.02
# p4a, eigenvalue4a = power(getGoogle(A, q), toleration)
# p4a = p4a / np.sum(p4a)
# print("4.4a:")
# print("difference ", p4a - p)
#
# q = 0.6
# p4b, eigenvalue4b = power(getGoogle(A, q), toleration)
# p4b = p4b / np.sum(p4b)
# print("4.4b:")
# print("difference ", p4b - p)


# 4.5

q = 0.15
A5 = A.copy()
A5[8-1, 11-1] = 3
A5[12-1, 11-1] = 3
p5, eigenvalue5 = power(get_google(A5, q), toleration)
p5 = p5 / np.sum(p5)
# print("\n4.5:")
# print("the new eigenvector: ", p5)
# for i in range(15):
#     print(i+1, ': ', p5[i])


# 4.6

n = 14
A6 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]])

q = 0.15
p6, eigenvalue6 = power(get_google(A6, q), toleration)
p6 = p6 / np.sum(p6)
print("\n4.6:")
print(p6)

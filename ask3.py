import numpy as np
import matplotlib.pyplot as plt


def place_max_to_driver(A, j, P, L):
    max1 = abs(A[j, j])
    pos = j
    for i in range(j+1, A.shape[0]):
        if abs(A[i, j]) > max1:
            max1 = abs(A[i, j])
            pos = i

    temp = np.array(A[j])
    A[j] = A[pos]
    A[pos] = temp

    temp = np.array(P[j])
    P[j] = P[pos]
    P[pos] = temp

    temp = np.array(L[j])
    L[j] = L[pos]
    L[pos] = temp


def palu(A):
    shape = A.shape[0]
    P = np.identity(shape)
    L = np.zeros((shape, shape), dtype=float)
    U = np.array(A, dtype=float)

    for j in range(shape-1):
        place_max_to_driver(U, j, P, L)
        for i in range(j+1, shape):
            if U[j, j] != 0:
                coefficient = U[i, j] / U[j, j]
            else:
                return P, L, U
            for col in range(j, shape):
                U[i, col] = U[i, col] - coefficient * U[j, col]
            L[i, j] = coefficient

    for i in range(shape):
        L[i, i] = 1.

    return P, L, U


def solve_system(A, b):
    shape = A.shape[0]
    P, L, U = palu(A)
    y = np.zeros(shape, dtype=float)
    z = np.dot(P, b)

    # solving the Ly = z system
    for i in range(shape):
        y[i] = z[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]

    # solving the Ux = y system
    x = np.zeros(shape, dtype=float)
    for i in range(shape-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, shape):
            x[i] -= U[i, j] * x[j]
        if U[i, i] != 0:
            x[i] /= U[i, i]
        else:  # if an element on the diagonal is zero, then the determinant of the upper triangle matrix
            return None   # is also zero, which that the system has either infinite or no solutions

    return x


B = np.array([[2, 1, 5],
              [4, 4, -4],
              [1, 3, 1]])

A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]])

C = np.array([[1, 1],
              [0, -7]])

b = np.array([5, 0, 6])

print(solve_system(B, b))

# P, L, U = palu(C)
# print(P)
# print(L)
# print(U)
#
#
# print("\n", np.dot(P, C), '\n', np.dot(L, U))



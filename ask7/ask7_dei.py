import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


# assisting functions
def place_max_to_pivot(A, j, P, L):
    """
    Swaps rows as needed such that the pivot element of the matrix A in the column j has the largest absolut value
    in its column. It also swaps the respective rows of the matrices P and L.
    :param A: the matrix
    :param j: the column
    :param P: the permutation matrix
    :param L: the lower triangular matrix
    """
    max1 = abs(A[j, j])
    pos = j
    for i in range(j+1, A.shape[0]):  # finds the row with the element that has the largest absolut value
        if abs(A[i, j]) > max1:
            max1 = abs(A[i, j])
            pos = i

    temp = np.array(A[j])  # swaps the rows of the matrix A
    A[j] = A[pos]
    A[pos] = temp

    temp = np.array(P[j])  # swaps the rows of the matrix P
    P[j] = P[pos]
    P[pos] = temp

    temp = np.array(L[j])  # swaps the rows of the matrix L
    L[j] = L[pos]
    L[pos] = temp


def palu(A):
    """
    Performs the PA=LU decomposition of the matrix A.
    :param A: the matrix
    :return: the permutation matrix P, the lower triangular matrix L and the upper triangular matrix U
    """
    shape = A.shape[0]
    P = np.identity(shape)
    L = np.zeros((shape, shape), dtype=np.float64)
    U = np.array(A, dtype=np.float64)  # A copy of the matrix A

    for j in range(shape-1):  # for every column
        place_max_to_pivot(U, j, P, L)
        for i in range(j+1, shape):  # for every row
            if U[j, j] != 0:  # If this element is zero, then the PA=LU decomposition is complete
                coefficient = U[i, j] / U[j, j]
            else:
                return P, L, U
            for col in range(j, shape):  # Performs the row operation R_k <- R_k + a*R_l
                U[i, col] = U[i, col] - coefficient * U[j, col]
            L[i, j] = coefficient

    for i in range(shape):  # fills the diagonal of L with ones
        L[i, i] = 1.

    return P, L, U


def solve_system(A, b):
    """
    Solves the linear system of equations Ax=b
    :param A: the coefficient matrix
    :param b: the vector b
    :return: the solution vector x
    """
    shape = A.shape[0]
    P, L, U = palu(A)
    y = np.zeros(shape, dtype=np.float64)
    z = np.dot(P, b)

    # solving the Ly = z system
    for i in range(shape):
        y[i] = z[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]

    # solving the Ux = y system
    x = np.zeros(shape, dtype=np.float64)
    for i in range(shape-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, shape):
            x[i] -= U[i, j] * x[j]
        if U[i, i] != 0:
            x[i] /= U[i, i]
        else:  # if an element on the diagonal is zero, then the determinant of the upper triangle matrix
            return None   # is also zero, which that the system has either infinite or no solutions

    return x


def least_squares_method(points, degree):
    """
    Calculates a polynomial of a certain degree based on a set of given points using the least squares method
    :param points:the points on the cartesian plane
    :param degree: the degree of the approximation polynomial
    :return: the calculated polynomial and the norm of the remainder vector b-Ax
    """
    n = len(points)
    A = np.ones((n, degree+1), dtype=np.float64)
    b = np.zeros((n, 1), dtype=np.float64)

    for i in range(n):  # makes the coefficient matrix A and the vector b
        x = points[i][0]
        for j in range(1, degree+1):
            A[i][j] = x
            x *= points[i][0]
        b[i] = points[i][1]

    C = np.dot(A.T, A)
    d = np.dot(A.T, b)
    x = solve_system(C, d)  # the solution vector x
    r = np.linalg.norm(b.T-np.dot(A, x), 2)  # the norm of the remainder vector b-Ax
    return Polynomial(x), r


points = ((0, 7.4650), (1, 7.4800), (2, 7.5950), (3, 7.7600), (4, 7.7400),
          (5, 7.9800), (6, 8.0000), (7, 8.1700), (8, 8.0500), (9, 8.1250))
approximation_polynomials = list()
for i in range(2, 5):
    approximation_polynomials.append(least_squares_method(points, i))

# plots the polynomials from the least squares method
t = np.linspace(0, 15, 100)
for i in range(2, 5):
    y = list()
    for x in t:
        y.append(approximation_polynomials[i - 2][0](x))
    plt.plot(t, y)

x_points = list(point[0] for point in points)
y_points = list(point[1] for point in points)
plt.scatter(x_points, y_points, marker='^', color='red')  # plots the known stock prices of the previous 10 days
plt.scatter((11, 15), (8.0450, 8.3), marker='^', color='red', edgecolors='black')  # plots the actual stock prices of days 11 and 15
plt.legend(['second degree', 'third degree', 'fourth degree'], loc='lower left')

# calculates the error for the first 10 days for each polynomial and stores it in a list
error_list = list()
for i in range(2, 5):
    error = list()
    for point in points:
        error.append(abs(point[1] - approximation_polynomials[i-2][0](point[0])))
    error_list.append(error)

print(f'Mean error (second degree): {np.mean(error_list[0]):.7f}')
print(f'Mean error (third degree): {np.mean(error_list[1]):.7f}')
print(f'Mean error (fourth degree): {np.mean(error_list[2]):.7f}')

print('\n21-2 forecast:')  # day 11
print(f'(second degree): {approximation_polynomials[0][0](11):.7f} error: {8.045-approximation_polynomials[0][0](11):.7f}')
print(f'(third degree): {approximation_polynomials[1][0](11):.7f} error: {8.045-approximation_polynomials[1][0](11):.7f}')
print(f'(fourth degree): {approximation_polynomials[2][0](11):.7f} error: {8.045-approximation_polynomials[2][0](11):.7f}')

print('\n28-2 forecast:')  # day 15
print(f'(second degree): {approximation_polynomials[0][0](15):.7f} error: {8.3-approximation_polynomials[0][0](15):.7f}')
print(f'(third degree): {approximation_polynomials[1][0](15):.7f} error: {8.3-approximation_polynomials[1][0](15):.7f}')
print(f'(fourth degree): {approximation_polynomials[2][0](15):.7f} error: {8.3-approximation_polynomials[2][0](15):.7f}')

plt.show()

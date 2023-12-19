import numpy as np
import matplotlib.pyplot as plt


# PA = LU

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
    L = np.zeros((shape, shape), dtype=float)
    U = np.array(A, dtype=float)  # A copy of the matrix A

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


# Cholesky

def cholesky(A):
    """
    It performs Cholesky decomposition on a symmetrical and positive definite matrix A and returns the lower triangular
    matrix L, which satisfies the equation A = LL^T.
    :param A: a symmetrical and positive definite matrix
    :return: the lower triangular matrix L of the Cholesky decomposition of the matrix A
    """
    shape = A.shape[0]
    A = np.array(A, dtype=float)
    R = np.zeros((shape, shape), dtype=float)
    for k in range(shape):
        if A[k, k] < 0:
            return None
        R[k, k] = np.sqrt(A[k, k])
        u = np.array([[A[k, j] / R[k, k] for j in range(k+1, shape)]])
        i = 0
        for j in range(k+1, shape):
            R[k, j] = u[0, i]
            i += 1

        uuT = np.dot(u.T, u)  # multiplies the column vector u.T with the row vector u
        row = 0
        for i in range(k+1, shape):
            col = 0
            for j in range(k+1, shape):
                A[i, j] = A[i, j] - uuT[row, col]
                col += 1
            row += 1

    return R.T


# Gauss-Seidel method

def gauss_seidel(A, b, x0, tol):
    """
    Solves the system of equations Ax=b using the Gauss-Seidel method
    :param A: a square matrix
    :param b: the vector b
    :param x0: an initial guess for the solution of the system
    :param tol: the toleration
    :return: the solution vector x
    """
    shape = A.shape[1]
    x = np.array(x0)
    norm = tol + 1

    while norm > tol:
        x0 = np.array(x)
        for i in range(shape):
            s1 = 0
            for j in range(i):
                s1 += A[i, j] * x[j]

            s2 = 0
            for j in range(i+1, shape):
                s2 += A[i, j] * x[j]

            x[i] = (b[i] - s1 - s2) / A[i, i]

        norm = infinite_norm(x - x0)

    return x


def infinite_norm(x):
    """
    Calculates the infinite norm of the vector x
    :param x: a vector
    :return: the infinite norm of the vector x
    """
    max1 = -1
    for i in range(x.shape[0]):
        element = abs(x[i])
        if element > max1:
            max1 = element
    return max1


# Gauss-Seidel for the 10x10 matrix A
n = 10
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = 5
    try:
        A[i+1, i] = -2
    except IndexError:
        pass
    try:
        A[i, i+1] = -2
    except IndexError:
        pass

b = np.ones(n, dtype=float)
b[0] = 3
b[9] = 3
x0 = np.zeros(n)

print(gauss_seidel(A, b, x0, 0.00005))

# Gauss-Seidel for the 10000x10000 matrix A
n = 10000
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = 5
    try:
        A[i+1, i] = -2
    except IndexError:
        pass
    try:
        A[i, i+1] = -2
    except IndexError:
        pass

b = np.ones(n, dtype=float)
b[0] = 3
b[9] = 3
x0 = np.zeros(n)

# print(gauss_seidel(A, b, x0, 0.00005))


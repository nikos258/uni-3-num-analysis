import numpy as np
from matplotlib import pyplot as plt


def power_method(A, k):
    """
    Implements the power method to find the eigenvalue with the largest absolut value and the respective eigenvector.
    :param A: the matrix
    :param k: the iteration of the algorithm
    :return: the eigenvector and the eigenvalue
    """
    shape = A.shape[0]
    x = np.ones(shape)  # creates a starting vector full of ones
    for i in range(k):
        u = x / np.linalg.norm(x)  # normalises the vector
        x = np.dot(A, u)
        l = np.dot(u, np.matrix(x).T)

    u = x / np.linalg.norm(x)  # normalises the vector
    return u, l


def get_google(A, q):
    """
    Calculates the Google matrix G based on the adjacency matrix A and the jumping probability q.
    :param A: the adjacency matrix
    :param q: the jumping probability q
    :return: the Google matrix G
    """
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

iterations = 50

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
p, eigenvalue = power_method(G, iterations)
p = p / np.sum(p)  # normalises the eigenvector p

print("4.2:")
print("The normalised eigenvector: ", p)
print("The eigenvalue: ", eigenvalue)


# 4.3

# makes a copy of the adjacency matrix A and changes some edges
A3 = A.copy()
A3[10-1, 1-1] = 1
A3[11-1, 1-1] = 1
A3[13-1, 1-1] = 1
A3[15-1, 1-1] = 1
A3[1-1, 2-1] = 0

q = 0.15
G3 = get_google(A3, q)
p3, eigenvalue3 = power_method(G3, iterations)
p3 = p3 / np.sum(p3)

print("\n4.3:")
print("The new eigenvector: ", p3)
print("The new eigenvalue: ", eigenvalue3)


# 4.4

q = 0.02
p4a, eigenvalue4a = power_method(get_google(A3, q), iterations)
p4a = p4a / np.sum(p4a)
print("\n4.4a:")
print("The new eigenvector: ", p4a)

q = 0.6
p4b, eigenvalue4b = power_method(get_google(A3, q), iterations)
p4b = p4b / np.sum(p4b)
print("\n4.4b:")
print("The new eigenvector: ", p4b)


# 4.5

# makes a copy of the adjacency matrix A and changes the values of some links
q = 0.15
A5 = A.copy()
A5[8-1, 11-1] = 3
A5[12-1, 11-1] = 3

p5, eigenvalue5 = power_method(get_google(A5, q), iterations)
p5 = p5 / np.sum(p5)
print("\n4.5:")
print("the new eigenvector", p5)


# 4.6

# makes the new adjacency matrix A which lacks the 10th page
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
p6, eigenvalue6 = power_method(get_google(A6, q), iterations)
p6 = p6 / np.sum(p6)
print("\n4.6:")
print(p6)

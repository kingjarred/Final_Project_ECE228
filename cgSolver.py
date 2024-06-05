'''
==============================================
Conjugate Gradient PDE Linear Numerical Solver
==============================================
''' 

import numpy as np

# function to solve a system of linear equations using the conjugate gradient method
def cgSolver(A, b, tol=1e-10, max_iter=1000):
    x, r, p = np.zeros_like(b), b - A @ np.zeros_like(b), b - A @ np.zeros_like(b)
    rs_old = np.dot(r.T, r)
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p.T, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r.T, r)
        if np.sqrt(rs_new) < tol: break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

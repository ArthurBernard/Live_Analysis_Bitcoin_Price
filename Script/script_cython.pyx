#! /usr/bin/env python3
import numpy as np
cimport numpy as np

# Arthur BERNARD

""" Some cython loops that run faster than python code """

#====================================================================================#

cpdef tuple MA_est_loop(np.ndarray[np.float64_t, ndim=2] u, 
        int q, int iterations, np.float64_t learning_rate):
    
    cdef np.ndarray[np.float64_t, ndim=2] a, theta
    
    (a, theta) = loop_3(u, q, iterations, learning_rate)
    return (a, theta)

cdef tuple loop_3(np.ndarray[np.float64_t, ndim=2] u, 
        int q, int iterations, np.float64_t learning_rate):
    
    cdef np.ndarray[np.float64_t, ndim=2] a, theta
    cdef int j, T, t2, t
    cdef np.float64_t c

    T = u.shape[0]
    theta = np.zeros([q, 1], dtype=np.float64) # init
    grad = np.zeros([q, 1], dtype=np.float64)
    for j in range(iterations):
        a = np.zeros([T, q+1], dtype=np.float64)
        a[0, 0] = u[0]
        a = loop_2(a, u, theta, T, q)
        for t2 in range(q):
            c = 0
            for t in range(T):
                c = c + a[t, t2+1]*a[t, 0]
            theta[t2, 0] = theta[t2, 0] - learning_rate*c
    return (a, theta)

cdef np.ndarray[np.float64_t, ndim=2] loop_2(np.ndarray[np.float64_t, ndim=2] a, 
        np.ndarray[np.float64_t, ndim=2] u, 
        np.ndarray[np.float64_t, ndim=2] theta, 
        int T, int q):
    
    cdef int t, t2
    cdef np.float64_t c
    
    for t in range(1, T):
        a = loop_1(a, t, q)
        c = 0
        for t2 in range(q):
            c = c + a[t, t2]*theta[t2, 0]
        a[t, 0] = u[t, 0] - c
        return a

cdef np.ndarray[np.float64_t, ndim=2] loop_1(np.ndarray[np.float64_t, ndim=2] a, 
        int t, int q):
    
    cdef int i
    
    for i in range(1, q+1):
        a[t, i] = a[t-1, i-1]
    return a

#====================================================================================#

cpdef np.ndarray[np.float64_t, ndim=2] auto_cov_cython(np.ndarray[np.float64_t, ndim=2] X, int h):
    return loop_1_aut_cov(X, h)

cdef np.ndarray[np.float64_t, ndim=2] loop_1_aut_cov(
        np.ndarray[np.float64_t, ndim=2] X, 
        int h):
    
    cdef int k, T, i
    cdef np.ndarray[np.float64_t, ndim=2] gama
    cdef np.float64_t x2bar, xbar_sup, xbar_inf
    
    T = X.shape[0]
    gama = np.zeros([h + 1, 1], dtype=np.float64)
    for k in range(h + 1):
        x2bar = 0
        xbar_inf = 0
        xbar_sup = 0
        for i in range(k, T):
            x2bar = x2bar + X[i, 0]*X[i-k, 0]
            xbar_inf = xbar_inf + X[i, 0]
            xbar_sup = xbar_sup + X[i-k, 0]
        gama[k, 0] = x2bar/(T - k) - (xbar_inf/(T - k))*(xbar_sup/(T - k))
    return gama

#====================================================================================#

cpdef np.ndarray[np.float64_t, ndim=2] auto_corr_cython(np.ndarray[np.float64_t, ndim=2] X, int h):
    return loop_auto_corr(X, h)

cdef np.ndarray[np.float64_t, ndim=2] loop_auto_corr(
        np.ndarray[np.float64_t, ndim=2] X, 
        int h):
    
    cdef int k, T, i
    cdef np.ndarray[np.float64_t, ndim=2] roh
    cdef np.float64_t x2bar, xbar
    
    T = X.shape[0]
    roh = loop_1_aut_cov(X, h)
    x2bar = 0
    xbar = 0
    for i in range(T):
        x2bar = x2bar + X[i, 0]*X[i, 0]
        xbar = xbar + X[i, 0]
    for k in range(h + 1):
        roh[k, 0] = roh[k, 0]/(x2bar/T - (xbar/T)*(xbar/T))
    return roh

#====================================================================================#

cpdef tuple AR_est_loop(
        np.ndarray[np.float64_t, ndim=2] y,
        np.ndarray[np.float64_t, ndim=2] X, 
        np.ndarray[np.float64_t, ndim=2] phi,
        int p, int q, int T):
    
    cdef int i, j, t
    cdef np.ndarray[np.float64_t, ndim=2] roh, roh_mat, u, mat_inv
    cdef np.float64_t m, cum
    
    for i in range(1, p):
        for t in range(i, T):
            X[t, i] = y[t - i, 0]
    roh = loop_auto_corr(y, h=p+q)
    roh_mat = np.ones([p+q, p+q], dtype=np.float64)
    for i in range(p+q):
        for j in range(i, p+q):
            roh_mat[j, i] = roh[j - i, 0]
            roh_mat[i, j] = roh[j - i, 0]
    mat_inv = np.linalg.inv(roh_mat[q:p+q, 0:p])
    for i in range(1, p+1):
        for j in range(p):
            phi[i, 0] = phi[i, 0] + mat_inv[i -1, j]*roh[j+1 + q, 0]
    m = 0
    cum = 0
    for t in range(T):
        m = m + y[t, 0]
    m = m/T
    for i in range(p):
        cum = cum + phi[i+1, 0]
    phi[0, 0] = m*(1 - cum)
    u = y
    for t in range(T):
        for i in range(p+1):
            u[t, 0] = u[t, 0] - X[t, i]*phi[i, 0]
    return (phi, u)

#====================================================================================#

cpdef np.ndarray[np.float64_t, ndim=2] MA_gp_loop(
        np.ndarray[np.float64_t, ndim=2] a, 
        np.ndarray[np.float64_t, ndim=2] u, 
        np.ndarray[np.float64_t, ndim=2] theta, 
        int T):
    
    cdef int t, q
    
    q = theta.shape[0] 
    for t in range(T):
        for i in range(q):
            u[t, 0] = u[t, 0] + a[t, i]*theta[i, 0]
    return u

#====================================================================================#

cpdef tuple AR_gp_loop(
        np.ndarray[np.float64_t, ndim=2] X, 
        np.ndarray[np.float64_t, ndim=2] res,
        np.ndarray[np.float64_t, ndim=2] phi,
        int T, int p):
    
    cdef int t, k
    cdef np.ndarray[np.float64_t, ndim=2] y
    cdef np.float64_t c
    
    y = np.zeros([T, 1], dtype=np.float64)
    for t in range(T):
        c = 0 
        for k in range(p+1):
            c = c + X[t, k]*phi[k, 0]
        y[t, 0] = c + res[t, 0]
        if t+1 < T:
            X[t+1, 1] = y[t, 0]
            for k in range(1, p):
                X[t+1, k+1] = X[t, k]
    return (y, X)

#====================================================================================#

cpdef np.float64_t mean_loop_cython(np.ndarray[np.float64_t, ndim=2] x):
    
    cdef np.float64_t m
    cdef int t, T

    T = x.size
    m = 0
    if x.shape[0] == T:
        for t in range(T):
            m += x[t, 0]
    elif x.shape[1] == T:
        for t in range(T):
            m += x[0, t]
    m /= T
    return m

#====================================================================================#

cpdef np.float64_t var_loop_cython(np.ndarray[np.float64_t, ndim=2] x):

    cdef np.float64_t s
    cdef int t, T

    T = x.size
    s = 0
    if x.shape[0] == T:
        for t in range(T):
            s += x[t, 0]**2
    if x.shape[1] == T:
        for t in range(T):
            s += x[0, t]**2
    s = s/T - (mean_loop_cython(x))**2
    return s
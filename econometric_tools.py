#! /usr/bin/env python3
import numpy as np
from script_cython import MA_est_loop, AR_est_loop, auto_cov_cython, auto_corr_cython, MA_gp_loop, AR_gp_loop, mean_loop_cython, var_loop_cython

# Arthur BERNARD

#============================================================================#
""" 
Define some tools:
- Generator process
- StatisticTools
- EconometricTools
- OptimizationTools
"""
#============================================================================#

class GeneratorProcess:
    """ 
    Class to generate process:
    
    AR generator process
    MA generator process
    ARMA generator process
    """
    def __init__(self, nb_obs, residue=None, X_init=0):
        """ 
        Initializing parameters:
        
        :nb_obs:  int of the number of observations
        :residue: np.array(T, 1) of the residue
        :X_init:  float of the first observation
        """
        self.T = nb_obs
        if residue.shape[1] == self.T:
            residue = residue.T
        self.res = residue
        self.X_init = X_init
    
    def AR_generator_process(self, phi, p, function=None):
        """ 
        Return the serie X_t following an auto regressive process 
        of order p such that: 
        X_t = phi_0 + phi_1*f(X_t-1) + ... + phi_p*f(X_t-p) + u_t
        By default f is the linear function: f(x) = x.
        
        :phi:      np.matrix(p+1, 1) coefficients of AR
        :p:        int order of the AR part
        :function: function default is a linear function
        """
        if phi.shape[0] == 1:
            phi = phi.T
        if phi.shape[0] == p:
            phi_0 = np.zeros([1, 1], dtype=np.float64)
            phi = np.concatenate((phi_0, phi))
        X = np.zeros([self.T, p+1], dtype=np.float64)
        X[0:self.T, 0:1] = np.ones([self.T, 1], dtype=np.float64)# Constant 
        if not function:
            X[0, 1] = self.X_init
            (self.y, self.X) = AR_gp_loop(X, self.res, phi, self.T, p)
        else:
            X_t = np.zeros([self.T, 1], dtype=np.float64)
            X[0, 1] = function(self.X_init)     # Initial observation
            for t in range(self.T): 
                X_t[t] = X[t]@phi + self.res[t] 
                if t+1 < self.T:
                    X[t+1, 1] = function(X_t[t])
                    if p > 1:
                        X[t+1, 2:p+1] = X[t, 1:p]
            self.y = X_t
            self.X = X
        return self
    
    def MA_generator_process(self, theta, q):
        """ 
        Return the serie u_t following an mobile average process 
        of order q such that: 
        u_t = a_t + theta_1*a_t-1 + ... + theta_p*a_t-q
        
        :theta: np.matrix(q,1) coefficients of MA
        :q:     int order of the MA
        """
        if theta.shape[0] == q:
            theta_0 = np.ones([1, 1], dtype=np.float64)
            theta = np.concatenate((theta_0, theta))
        a = np.zeros([self.T, q+1], dtype=np.float64)
        u = np.zeros([self.T, 1], dtype=np.float64)
        for i in range(q+1):
            a[i:self.T, i] = self.res[0:self.T-i, 0]
        self.u = MA_gp_loop(a, u, theta, self.T)
        self.a = a
        return self
    
    def ARMA_generator_process(self, phi, theta, p, 
                               q, function=None):
        """ 
        Return the serie X_t following an auto regressive and an 
        mobile average process of order p, q such that: 
        X_t = phi_0 + phi_1*f(X_t-1) + ... + phi_p*f(X_t-p) + u_t
        u_t = a_t + theta_1*a_t-1 + ... + theta_p*a_t-q
        
        :phi:   np.matrix(p+1, 1) coefficients of AR
        :theta: np.matrix(q, 1)   coefficients of MA
        :p:     int order of the AR
        :q:     int order of the MA
        """
        self.res = self.MA_generator_process(theta, q).u
        return self.AR_generator_process(phi, p, function)
    
#============================================================================#
    
class StatisticTools:
    """ 
    Class with some statistic tools: 
    
    Mean
    Variance
    Standard deviation
    """
    def __init__(self):
        pass

    def mean(self, x):
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            return mean_loop_cython(x)
        else:
            print('not use cython')
            t = len(x)
            return sum(x)/t

    def var(self, x):
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            return var_loop_cython(x)
        else:
            print('not use cython')
            t = len(x)
            s = 0
            for i in range(t):
                s += x[i]**2
            return s/t - (sum(x)/t)**2

    def std(self, x):
        return self.var(x)**0.5

#============================================================================#
    
class EconometricTools:
    """ 
    Class with some econometric tools:
    
    Auto-covariance vector
    Auto-Regressive of order p estimation
    Moving-Avergae of order q estimation
    ARMA of order p, q estimation
    Non-parametric estimation
    Gaussian kernel
    """
    def __init__(self):
        pass
        
    def auto_cov_emp_vect(self, X, h=50):
        """ 
        Return autocovariance vector 
        
        :X: np.matrix(T,1)
        :h: int as the number of lag, default is 50
        """
        return auto_cov_cython(X, h)

    def auto_corr_emp_vect(self, X, h=50):
        """ 
        Return autocorrelation vector 
        
        :X: np.matrix(T,1)
        :h: int as the number of lag, default is 50
        """
        if np.var(X) != 0:
            return auto_corr_cython(X, h)
        else:
            return auto_cov_cython(X, h)
    
    def AR_est(self, X, p, q=0):
        """
        Return the estimate coefficients of AR(p) process by the 
        Yule-Walker estimation method:
        
        :X: np.matrix(T,1)
        :p: int of order of the AR part
        :q: int of order of the MA part if ARMA, default is 0
        """
        T = X.shape[0]
        y = X
        one = np.ones([T, 1], dtype=np.float64)
        X = np.zeros([T, p+1], dtype=np.float64)
        u = np.zeros([T, 1], dtype=np.float64)
        phi = np.zeros([p + 1, 1], dtype=np.float64)
        X[0:T, 0:1] = one
        for i in range(1, p):
            X[i:T,i] = y[0:T-i, 0]
        sig2 = np.var(y, dtype=np.float64)
        if sig2 == 0:
            return phi, u
        roh = self.auto_cov_emp_vect(y, p+q) / sig2
        roh_mat = np.ones([p+q, p+q], dtype=np.float64)# init auto-corr
        for i in range(p+q):
            roh_mat[i:p+q, i] = roh[0:p+q-i, 0]
            roh_mat[i, i:p+q] = roh[0:p+q-i, 0].T
        try:
            phi[1:p+1,0] = np.linalg.pinv(roh_mat[q:p+q,0:p])@roh[q+1:q+p+1,0]
        except np.linalg.linalg.LinAlgError:
            print('Matrix not pseudo-inversible in the AR est')
            return phi, u
        phi[0] = (one.T@y/T)*(1 - sum(phi[1:p+1, 0]))
        u = y - X@phi
        return phi, u

    def MA_est(self, u, q, iterations=20, learning_rate=0.01):
        """ 
        Return the estimate coefficients of MA(q) process by the 
        minimzation of the square error with method the descent 
        of gradient:
        
        :u:             np.matrix(T, 1) of the serie observed
        :q:             int of order of the MA part
        :iterations:    int of iterations, default is 20
        :learning_rate: float of the learning rate, default 0.01
        
        Remark: gradient descent method is not the best way to 
        MA(q) estimation.
        """
        if q == 0:
            return np.array([]), u
        T = u.shape[0]
        (a, theta) = MA_est_loop(u, q, iterations, learning_rate)
        return theta, a[0:T, 0:1]
    
    def ARMA_est(self, X, p, q, iterations=20, learning_rate=0.01):
        """
        Return the estimate coefficient of an ARMA(p,q) process. 
        See AR_est and MA_est to know which methods are used.
        
        :X: np.array(T,1) 
        :p: int of the order of AR part
        :q: int of the order of MA part
        
        Remark: The estimation of the MA part can be slow.
        """
        phi, u = self.AR_est(X, p, q)
        if u.all() == 0:
            return phi, None, u, None
        theta, a = self.MA_est(u, q, 
            iterations=iterations, 
            learning_rate=learning_rate)
        return phi, theta, u, a

    def best_ARMA(self, X, pmax=5, qmax=5, iterations=10, learning_rate=0.001):
        """
        Return the order of the best ARMA model to minimize the AIC
        
        :X: np.array(T, 1)
        :pmax: max of AR order
        :qmax: max of MA order
        :iterations: int of the max iterations for the optimization method
        :learning_rate: float of learning rate for the optimization method
        """
        if isinstance(X, list):
            X = np.array([X]).T
        T = np.shape(X)[0]
        best_p = 0
        best_q = 0
        AIC = 10000000000
        for p in range(1, pmax):
            for q in range(qmax):
                phi_est, theta_est, u, a = (self.ARMA_est(X, p, q, 
                    iterations=iterations, learning_rate=learning_rate))
                if u.all() == 0:
                    pass
                elif T*np.log(u.T@u/T) + 2*(p+q+1) < AIC:
                    best_p = p
                    best_q = q
                    AIC = T*np.log(u.T@u/T) + 2*(p+q+1)
        return (best_p, best_q)
    
    def non_para_est(self, X, y):
        """
        Return the non-parametric estimation by the Nadaraya-
        Watson method with silvermann bandwidth and gaussian 
        kernel:
        
        :X: np.array(T, 1)
        :y: np.array(T, 1)
        """
        T = y.shape[0]
        one = np.ones([1, T])
        x = np.linspace(min(X), max(X), num=T, dtype=np.float64)
        h = 1.06*np.std(y)*T**(-0.2)  # h of silvermann
        est = np.zeros([T, 1], dtype=np.float64)
        m = one@X / T
        s = np.std(X, dtype=np.float64)
        yt = y.T
        for i in range(T):
            ker = self.ker_norm((x[i] - X)/h)
            est[i, 0] = (yt@ker)/(one@ker)
        return est
    
    def ker_norm(self, x, m=0, s=1):
        """ Gaussian kernel:
        
        :x: np.array(T, 1)
        :m: float default is 0
        :s: float default is 1
        """
        stand = (x - m)/s
        pi = 3.141592653589793
        return np.exp(-(stand*stand)/2, dtype=np.float64)/(s*(2*pi)**(0.5))

    def kernel_density(self, X):
        """
        Return the estimation of the density by the Parzen-Rosenblatt method
        with silvermann bandwidth and gaussian kernel:

        :X: np.array(T, 1)
        """
        T = np.shape(X)[0]
        x = np.linspace(min(X), max(X), num=T)
        est = np.zeros([T, 1], dtype=np.float64)
        h = 1.06*np.std(X)*T**(-0.2)
        if h == 0:
            return est
        for i in range(T):
            est[i] = np.ones([1, T])@self.ker_norm((x[i] - X)/h)/(T*h)
        return est
    
#============================================================================#
    
class OptimizationTools:
    """ 
    Class with some optimization tools:
    
    Gradient descent method
    """
    def __init__(self, function, theta_init, gradient=None, 
                 hessien=None, learning_rate=0.1, 
                 regularization=0.1, iterations=200):
        """ 
        Parametization:
        
        :function:   function to optimize
        :theta_init: np.matrix(k, 1) of starting parameters
        :gradient:   np.matrix(k, 1) of the gradient function
        :hessien:    np.matrix(k, k) of the hessien function
        :alpha:      float learning rate to the gradient descent
        :lamb:       float of coefficient of the regularization
        :it:         int of the number of iterations
        """
        self.alpha = learning_rate
        self.lamb = regularization
        self.it = iterations
        self.func = function
        self.theta = theta_init
        self.grad = gradient
        self.hess = hessien
    
    def gradient_descent(self):
        """ 
        Algorithm of the gradient descent, return parameters which minimize
        the function. 
        """
        alpha = self.alpha
        cost = self.func(self.theta)
        for i in range(self.it):
            grad_theta = self.grad(self.theta).T
            self.theta = self.theta - alpha*grad_theta
            new_cost = self.func(self.theta)
            if cost - new_cost < 0:
                print('Algo not converging on the {}th iteration.'.format(i))
                alpha = alpha/10
            elif cost - new_cost < 0.0001:
                print('Stop iteration at the {}th'.format(i))
                return self.theta
            else:
                alpha = alpha*(1 + 3/(i + 1)) 
            cost = new_cost
        return self.theta
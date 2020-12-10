#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:16:13 2020

@author: Manuel Sánchez Torrón
"""

import numpy as np

# from sage.all import *
# from sage.stats.distributions.discrete_gaussian_integer\
# import DiscreteGaussianDistributionIntegerSampler

""""Given x in [0, q], returns x with values in [(-q-1)/2. (q-1)/2]"""


def balance(x, q):
    x %= q
    return np.array([n if n <= (q-1)/2
                     else n-q for n in x.flatten()]).reshape(np.shape(x))


"""
Rejection sampling algorithm
If parameters chosen as BBC+18 (ie B small), output is 1 with probability 1/rho
"""


def Rej(Z, B, sigma, rho):
    u = np.random.random()
    if u > 1/rho * np.exp((-2*Z.flatten().dot(B.flatten())
                           + np.linalg.norm(B.flatten())**2)/(2*sigma**2)):
        return 0
    else:
        return 1


"""
Gaussian sample
Given sigma and a shape, returns an np array chosen from the gaussian
distribution (kind of) with parameter sigma with the given shape
"""


def gaussian_sample(sigma, shape):
    # D = DiscreteGaussianDistributionIntegerSampler(sigma)
    # return np.array([D() for _ in range(np.prod(shape))]).reshape(shape)
    samples = np.random.normal(0, sigma, np.prod(shape))
    round_samples = np.array([int(np.floor(s)) for s in samples])
    return round_samples.reshape(shape)


"""
Evaluation of polynomial p in point x, where p may have matrix coefficients
Polynomial p is an array where p[i] = coeff. of X^i
"""


def polyeval(p, x):
    part_sum = np.zeros_like(p[0])
    pow_x = np.eye(x.shape[0], dtype=int)
    for coef in p:
        part_sum += pow_x @ coef
        pow_x = pow_x * x
    return part_sum


"""
Evaluation of polynomial p in point x, where p may have matrix coefficients
The powers of x are reduced mod q on each iteration
"""


def polymodeval(p, x, q):
    part_sum = np.zeros_like(p[0])
    pow_x = np.eye(x.shape[0], dtype=int)
    for coef in p:
        part_sum += pow_x @ coef
        pow_x = (pow_x * x) % q
    return part_sum



"""
Multiplication of two polynomials that may have matrix coefficients
Multiplication is element-wise
"""


def matpolymul(p1, p2):
    number_coefficients = p1.shape[0]+p2.shape[0]-1
    res = np.array([np.zeros_like(p1[0]) for _ in range(number_coefficients)])
    for i in range(p1.shape[0]):
        for j in range(p2.shape[0]):
            res[i+j] += p1[i] * p2[j]
    return res



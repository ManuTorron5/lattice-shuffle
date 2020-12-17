#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:53:05 2020

@author: Manuel Sánchez Torrón
"""

import time
import math
import sympy as sp
import numpy as np
from random import randint
from util import balance, gaussian_sample


"""
Commitment scheme
Given the security parameter lambda, the modulo of the messages and
the total elements of the message, generates a commitment key
"""


def generateCommitmentKey(lamb, p, N):
    # Set parameters as in BBC+18
    # P = self.n*self.k**2*self.m**2*self.p**2
    # B = P*self.N  # B = O(PN)
    # Pp = B*P
    r = int(np.ceil(np.log(N)))  # r = O(log n) or O(log N)?
    q = sp.nextprime(2**100)  # q = sp.nextprime(Pp*math.sqrt(r)) !!!
    # print("P = " + str(P))
    # print("B = " + str(B))
    # print("Pp = " + str(Pp))
    # print("r = " + str(r))
    # print("log2 q = " + str(math.log2(q)))

    n = int(np.ceil(np.sqrt(N*r*math.log(q, p)/lamb*np.log2(N*lamb*p))))
    k = int(np.ceil(lamb/np.log2(p)))
    m = int(np.ceil(N/(n*k)))

    # print("n = " + str(n))
    # print("k = " + str(k))
    # print("m = " + str(m))

    nprime = int(2*r*math.log(q, p))

    A1 = np.array([randint(-(q-1)//2, (q-1)//2) for _ in range(r*nprime)])\
        .reshape(r, nprime)
    A2 = np.array([randint(-(q-1)//2, (q-1)//2) for _ in range(r*n)])\
        .reshape(r, n)
    # print("Shape of matrix A1: " + str(A1.shape))
    # print("Shape of matrix A2: " + str(A2.shape))
    return [p, q, r, n, k, m, A1, A2]


"""
Commitment method
Given a message and randomness with shape given by the commitment key
returns the commitment of the message
"""


def commit(msg, randomness, ck):  # r = randomness
    # p = ck[0]
    q = ck[1]
    # r = ck[2]
    # n = ck[3]
    # k = ck[4]
    A1 = ck[6]
    A2 = ck[7]

    # nprime = int(2*r*math.log(q, p))
    # msg = np.resize(msg, (k, n))
    # randomness = np.resize(randomness, (k, nprime))
    C = (A1 @ randomness[0] % q) + (A2 @ msg[0] % q)
    for i in range(msg.shape[0]-1):
        C = np.vstack((C, (A1 @ randomness[i+1]) % q + (A2 @ msg[i+1]) % q))
    return balance(C, q)
    # return C


def commit_test():

    # Commitment for m messages in (Z_p)^kxn
    # Commits to each (kxn)-matrix as BBC+18 page 26 (k instead of 2k)
    # In the arithmetic circuit protocol we will commit to m (2kxn)-matrices

    lamb = 128
    p = 4093
    N = 100000

    # n = int(np.ceil(np.sqrt(N)))
    # k = int(N/n)  # int(np.ceil(lamb/np.log2(p)))
    # m = int(np.ceil(N/(k*n)))

    msg = np.random.randint((-p+1)/2, (p-1)/2+1, size=N)

    print("Messages: " + str(msg) + " (modulo " + str(p) + ")")

    start_time = time.time()

    ck = generateCommitmentKey(lamb, p, N)
    p, q, r, n, k, m = ck[0:6]

    nprime = int(2*r*math.log(q, p))

    sigma1 = 48*np.sqrt(k*n)*k*m*p**2

    msg = np.resize(msg, (m, k, n))
    randomness = gaussian_sample(sigma1, (m, k, nprime))

    com_msg = np.array([commit(msg[i], randomness[i], ck) for i in range(m)])

    end_time = time.time()

    # print("Committed messages:", com_msg)

    print("Size of the message:", msg.size, "elements modulo", p)
    print("Size of the commitment:", com_msg.size, "elements modulo ", ck[1])

    print("Total execution time:", end_time - start_time, "seconds")


#commit_test()

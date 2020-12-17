#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:22:02 2020

@author: Manuel Sánchez Torrón
"""

import time
import math
import numpy as np
import sympy as sp
from util import Rej, gaussian_sample


class Prover:

    A = None
    S = None
    T = None
    sigma = None
    Y = None

    def __init__(self, lamb, q, A, S, T):
        self.q = q
        self.A = A
        self.S = S
        self.T = T
        self.Y = []

        s = np.linalg.norm(S, 2) + 1  # Upper bound on s_1(S)

        self.ell = S.shape[1]
        self.n = lamb + 2
        assert A.shape[1] == S.shape[0]
        self.v = A.shape[1]

        self.rho = 4
        self.sigma = 12/np.log(self.rho)*s*np.sqrt(self.ell*self.n)
        self.B = math.sqrt(2*self.v)*self.sigma
        # This B is somehow known by the verifier

        # print("s = " + str(s))
        # print("sigma = " + str(self.sigma))
        # print("B = " + str(self.B))

    def calculateW(self):
        self.Y = gaussian_sample(self.sigma, (self.v, self.n))
        W = np.matmul(self.A, self.Y)
        return W

    def calculateZ(self, C):
        self.C = C
        self.Z = np.matmul(self.S, self.C) + self.Y
        return self.Z

    def reject(self):
        return Rej(self.Z, self.S@self.C, self.sigma, self.rho)


class Verifier:

    A = None
    T = None
    W = None
    C = None

    def __init__(self, lamb, q, A, T):
        self.lamb = lamb
        self.q = q
        self.A = A
        self.T = T
        self.W = []
        self.C = []

    def calculateC(self, W):
        self.W = W
        n = W.shape[1]
        ell = self.T.shape[1]
        self.C = np.random.randint(2, size=[ell, n])
        return self.C

    def verify(self, Z, B):
        AZ = self.A @ Z
        TC = self.T @ self.C
        return np.array_equal(AZ % self.q, (TC + self.W) % self.q)\
            and np.all(np.linalg.norm(Z, np.inf, axis=0) <= B)


class Proof:

    prover = None
    verifier = None

    W = None
    C = None
    Z = None

    bit = None
    num_aborts = None
    running_time = None
    proof_size = None

    def __init__(self, lamb, q, A, S, T):
        self.prover = Prover(lamb, q, A, S, T)
        self.verifier = Verifier(lamb, q, A, T)

    def run(self):
        abort = True
        self.num_aborts = 0

        # Protocol starts

        start_time = time.time()
        self.proof_size = 0

        self.W = self.prover.calculateW()

        self.proof_size += self.W.size

        while abort:

            self.C = self.verifier.calculateC(self.W)
            self.Z = self.prover.calculateZ(self.C)

            # self.proof_size += self.Z.size
            # self.proof_size += self.Z.nbytes

            # Rejection sampling
            abort = self.prover.reject()
            self.num_aborts += abort

        self.proof_size += self.Z.size

        # Verification
        B = self.prover.B  # ???

        self.bit = self.verifier.verify(self.Z, B)

        end_time = time.time()
        self.running_time = end_time - start_time

        return self.bit, self.proof_size

    def print_stats(self):

        # print(self.A)
        # print(self.S)
        # print(self.T)

        print("Verification:", self.bit)
        print("Times aborted:", self.num_aborts)
        print("Total execution time:", self.running_time, "seconds")
        print("Size of the witness:", self.prover.S.size, "elements")
        print("Size of communication:", self.proof_size,
              "elements modulo", self.prover.q)
        # print("Size of the witness: " +
        # str(self.prover.S.nbytes / 1000) + " kB")
        # print("Size of the proof: " + str(self.proof_size / 1000) + " kB")


def ZKP_test():
    lamb = 128  # Security parameter lambda
    q = sp.nextprime(4093)  # Prime for base field Z_q

    # Invent matrices A, S, T
    ell = 1000  # Number of equations
    r = 100  # poly(lambda)
    v = 10  # poly(lambda)
    # Parameters r, v and l should be chosen by the commitment given N = l*r*v

    A = np.random.randint(0, q, size=[r, v])
    S = np.random.randint(-1, 2, size=[v, ell])
    T = (A @ S) % q

    # print(A)
    # print(S)
    # print(T)

    proof = Proof(lamb, q, A, S, T)
    proof.run()
    proof.print_stats()


#ZKP_test()

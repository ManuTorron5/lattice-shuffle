#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:37:57 2020

@author: Manuel Sánchez Torrón
"""
import time
import random
import math
import numpy as np
from scipy import linalg
from random import randint
from util import Rej, gaussian_sample, polyeval, polymodeval, matpolymul, balance
# from siszkp import Proof
from commitmentScheme import generateCommitmentKey, commit
# from commitAndProve import commit_and_prove


class LinearConstraintArgument(object):

    # Parameters: security level lambda, prime p and N triples ai, bi, ci
    # such that ai*bi = ci (flatA = [a1, a2, ..., aN])

    lamb = None
    p = None
    aij = None
    bij = None
    cij = None
    wuaij = None
    wubij = None
    wucij = None

    def __init__(self, lamb, p, aij, bij, cij, wuaij, wubij, wucij):
        self.lamb = lamb
        self.p = p
        self.aij = aij
        self.bij = bij
        self.cij = cij
        self.wuaij = wuaij
        self.wubij = wubij
        self.wucij = wucij

    def run(self):
        assert self.aij.size == self.bij.size == self.cij.size
        N = self.aij.size

        # Generates the commitment scheme for N values in Zp with security
        # lambda. This defines n, k, m, q ... (?)
        ck = generateCommitmentKey(self.lamb, self.p, N)

        m = ck[5]
        k = ck[4]
        n = ck[3]

        # print(ck)

        Ai = np.resize(self.aij, (m, k, n))
        Bi = np.resize(self.bij, (m, k, n))
        Ci = np.resize(self.cij, (m, k, n))

        # Create prover
        prover = LinearConstraintArgumentProver\
            (self.lamb, self.p, Ai, Bi, Ci, self.wuaij, self.wubij, self.wucij, ck)

        # Create verifier
        verifier = LinearConstraintArgumentVerifier(self.lamb, self.p, ck)

        # Start protocol
        start_time = time.time()
        proof_size = 0

        bfA0, bfAi, bfB0, bfBi, bfC0, bfCi = prover.commit_witness()

        proof_size += bfA0.size
        proof_size += bfAi.size
        proof_size += bfB0.size
        proof_size += bfBi.size
        proof_size += bfC0.size
        proof_size += bfCi.size

        bfy = verifier.generate_challenge()

        print("\nThe verifier sends the challenge y = " + str(bfy))

        bfHl = prover.compute_and_commit_polynomials(bfy)

        proof_size += bfHl.size

        bfx = verifier.generate_challenge()

        print("\nThe verifier sends the challenge x = " + str(bfx))

        prover.evaluate_polynomials(bfy, bfx)

        bfD, bfE = prover.modulus_correction(bfy, bfx)

        proof_size += bfD.size
        proof_size += bfE.size

        bfz = verifier.generate_challenge()

        print("\nThe verifier sends the challenge z = " + str(bfz))

        A, alpha, B, beta, rho, barD, bardelta\
            = prover.rejections(bfy, bfx, bfz)

        proof_size += A.size
        proof_size += alpha.size
        proof_size += B.size
        proof_size += beta.size
        proof_size += rho.size
        proof_size += barD.size
        proof_size += bardelta.size


        bit = verifier.verify(bfA0, bfAi, bfBi, bfBmm1, bfCi, bfHl, bfD, bfE,
                              A, alpha, B, beta, rho, barD, bardelta,
                              bfy, bfx, bfz)

        end_time = time.time()

        total_time = end_time - start_time

        return bit, total_time, proof_size


class LinearConstraintArgumentProver(object):
    lamb = None
    ck = None
    Ai = None
    Bi = None
    Ci = None
    k = None
    m = None
    n = None
    N = None
    p = None
    q = None
    nprime = None
    sigma1 = None
    sigma2 = None
    sigma3 = None
    sigma4 = None

    def __init__(self, lamb, p, Ai, Bi, Ci, wuaij, wubij, wucij, ck):

        assert Ai.shape == Bi.shape == Ci.shape

        self.Ai = Ai
        self.Bi = Bi
        self.Ci = Ci
        
        self.wuaij = wuaij
        self.wubij = wubij
        self.wucij = wucij

        # print("Ai = " + str(Ai))
        # print("\nBi = " + str(Bi))
        # print("\nCi = " + str(Ci))

        self.ck = ck
        self.p = ck[0]
        self.q = ck[1]
        self.n = ck[3]
        self.k = ck[4]
        self.m = ck[5]
        self.nprime = int(2*ck[2]*math.log(ck[1], ck[0]))

        self.sigma1 = 48*math.sqrt(self.k*self.n)*self.k*self.m*self.p**2
        self.sigma2 =\
            24*math.sqrt(2*self.k*self.n)*self.k*self.p**2*(2*self.m+1)
        self.sigma3 = 24*math.sqrt(2*self.k*self.n)\
            *(1 + self.k + 2*self.m*self.k*self.p)
        self.sigma4 = 24*math.sqrt(2*self.k*self.n)*(2*self.m+1)\
            *self.k*self.p**2

        # print("\nsigma1 = " + str(self.sigma1))
        # print("\nsigma2 = " + str(self.sigma2))
        # print("\nsigma3 = " + str(self.sigma3))
        # print("\nsigma4 = " + str(self.sigma4))

    def commit_witness(self):
        

        self.A0 = gaussian_sample(self.sigma1, (2*self.k, self.n))
        self.B0 = gaussian_sample(self.sigma1, (2*self.k, self.n))
        self.C0 = gaussian_sample(self.sigma1, (2*self.k, self.n))

        self.alphai = np.random.randint(0, self.p,
                                        [self.m, self.k, self.nprime])
        self.betai = np.random.randint(0, self.p,
                                       [self.m, self.k, self.nprime])
        self.gammai = np.random.randint(0, self.p,
                                        [self.m, 2*self.k, self.nprime])

        
        self.alpha0 = gaussian_sample(self.sigma1, (2*self.k, self.nprime))
        self.beta0 = gaussian_sample(self.sigma1, (2*self.k, self.nprime))
        self.gamma0 = gaussian_sample(self.sigma1, (2*self.k, self.nprime))
        


        self.bfAi = np.array(
            [commit(np.vstack((self.Ai[i], np.zeros_like(self.Ai[i], dtype=int))),
                    np.vstack((self.alphai[i], np.zeros_like(self.alphai[i], dtype=int))), self.ck)
             for i in range(self.m)])
        self.bfBi = np.array(
            [commit(np.vstack((self.Bi[i], np.zeros_like(self.Bi[i], dtype=int))),
                    np.vstack((self.betai[i], np.zeros_like(self.betai[i], dtype=int))), self.ck)
             for i in range(self.m)])
        self.bfCi = np.array(
            [commit(np.vstack((self.Ci[i], np.zeros_like(self.Ci[i], dtype=int))),
                    np.vstack((self.gammai[i], np.zeros_like(self.gammai[i], dtype=int))), self.ck)
             for i in range(self.m)])
        

        # print("\nbfAi = " + str(self.bfAi))
        # print("\nbfBi = " + str(self.bfBi))
        # print("\nbfCi = " + str(self.bfCi))

        self.bfA0 = commit(self.A0, self.alpha0, self.ck)
        self.bfB0 = commit(self.B0, self.beta0, self.ck)
        self.bfC0 = commit(self.C0, self.gamma0, self.ck)
        

        # print("\nbfA0 = " + str(self.bfA0))
        # print("\nbfBmm1 = " + str(self.bfBmm1))

        return self.bfA0, self.bfAi, self.bfB0, self.bfBi, self.bfC0, self.bfCi

    def compute_and_commit_polynomials(self, bfy):
        
        Wuai = np.array([np.vstack((self.wuaij[u], np.zeros((self.k,self.n),dtype=int))) for u in range(U)])
        
        
    def evaluate_polynomials(self, bfx, bfy):
        
    def modulus_correction(self, bfy, bfx):
        
    def rejections(self, bfy, bfx, bfz):
        


class LinearConstraintArgumentVerifier(object):
    lamb = None
    ck = None
    calB = None
    k = None
    m = None
    n = None
    N = None
    p = None
    q = None
    np = None

    def __init__(self, lamb, p, m, k, n, ck):

        self.lamb = lamb
        self.p = p
        self.m = m
        self.k = k
        self.n = n
        self.ck = ck

        self.q = ck[1]

        self.sigma1 = 48*math.sqrt(self.k*self.n)*self.k*self.m*self.p**2
        self.sigma2 =\
            24*math.sqrt(2*self.k*self.n)*self.k*self.p**2*(2*self.m+1)
        self.sigma3 = 24*math.sqrt(2*self.k*self.n)\
            *(1 + self.k + 2*self.m*self.k*self.p)
        self.sigma4 = 24*math.sqrt(2*self.k*self.n)*(2*self.m+1)\
            *self.k*self.p**2

    # Generate challenges bfy, bfx and bfz
    def generate_challenge(self):
        c = np.random.randint(0, self.p, 2*self.k)
        return c

    def verify(self, bfA0, bfAi, bfBi, bfBmm1, bfCi, bfHl, bfD, bfE,
               A, alpha, B, beta, rho, barD, bardelta, bfy, bfx, bfz):


def linearConstraintArgument_test():

    lamb = 128
    p = 127
    N = 1000

    np.set_printoptions(threshold=5)
    
    argument = LinearConstraintArgument(lamb, p, aij, bij, cij, wuaij, wubij, wucij)
    print(argument.run())



linearConstraintArgument_test()

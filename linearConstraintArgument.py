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
# from commitAndProve import commit_and_prove_all


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
        Mbfx = np.diag(bfx)
        Mbfy = np.diag(bfy)
        powered_Mbfy = Mbfy
        powered_Mbfx = Mbfx

        self.A = self.A0
        self.alpha = self.alpha0
        self.B = self.Bmm1
        self.beta = self.betamm1
        # Calculate value A = A0 + Aizeros[0]*y*x + ... + Aizeros[m-1]*y^m*x^m
        # and alpha = alpha0 + ...
        for i in range(self.m):
            Mxy = (powered_Mbfx @ powered_Mbfy) % self.p
            self.A += Mxy @ self.Aizeros[i]
            self.alpha += Mxy @ self.alphaizeros[i]
            self.B += (powered_Mbfx % self.p) @ self.Bizeros[self.m-i-1]
            self.beta += (powered_Mbfx % self.p) @ self.betaizeros[self.m-i-1]
            powered_Mbfy = powered_Mbfy @ Mbfy
            powered_Mbfx = powered_Mbfx @ Mbfx

        self.A = self.A % self.p
        self.B = self.B % self.p
        # print("\nA = " + str(self.A))
        # print("\nB = " + str(self.B))

        self.Ax = polymodeval(self.AX, Mbfx, self.p)
        self.Bx = polymodeval(self.BX, Mbfx, self.p)

        # print("\nA(x) = " + str(self.Ax))
        # print("\nB(x) = " + str(self.Bx))

        # Check: A(x) = A mod p, B(x) = B mod p
        assert (self.Ax % self.p == self.A % self.p).all
        assert (self.Bx % self.p == self.B % self.p).all

    def modulus_correction(self, bfy, bfx):
        Mbfy = np.diag(bfy)
        Mbfx = np.diag(bfx)
        self.D = (self.A * self.B) % self.p
        - np.sum([(Mbfy**(i+1)%self.p) @ self.CiCpi[i]
                 for i in range(self.m)], axis=0)
        - polymodeval(self.Hl, Mbfx, self.p)
        + (Mbfx % self.p)**(self.m + 1) @ self.Hl[self.m]

        self.delta = gaussian_sample(self.sigma2, (2*self.k, self.nprime))
        self.bfD = commit(self.D, self.delta, self.ck)
        self.E = self.p * gaussian_sample(self.sigma3, (2*self.k, self.n))
        self.epsilon = gaussian_sample(self.sigma4, (2*self.k, self.nprime))
        self.bfE = commit(self.E, self.epsilon, self.ck)

        # print("\nD = " + str(self.D))
        # print("\nE = " + str(self.E))

        assert (self.E % self.p == 0).all

        return self.bfD, self.bfE

    def rejections(self, bfy, bfx, bfz):
        Mbfy = np.diag(bfy)
        Mbfx = np.diag(bfx)
        Mbfz = np.diag(bfz)

        M1 = np.hstack((self.A, self.alpha, self.B, self.beta))
        # print("\nAalphaBbeta = " + str(M1))
        M2 = np.hstack((self.A0, self.alpha0, self.Bmm1, self.betamm1))
        # print("\nA0alpha0B0beta0 = " + str(M2))
        self.e = 2
        # print(Rej(M1, M1 - M2, self.sigma1, self.e)) # What is e??
        # Pass all rejections with probability 1/e⁴

        # Calculate rho
        # First summand
        self.rho = 0
        for i in range(self.m):
            self.rho += ((Mbfx**(self.m+1) @ Mbfy**i) % self.p) @ self.gammai[i]
            #  The m+1 might be m+1-i

        # Second summand
        self.rho += polymodeval(self.etal, Mbfx, self.p) -\
            (Mbfx**(self.m + 1) % self.p) @ self.etal[self.m]
        # Third summand
        self.rho += self.delta

        # print("\nrho = " + str(self.rho))
        # print(Rej(self.rho, self.rho - self.delta, self.sigma2, self.e))

        self.barD = (Mbfz % self.p) @ self.D + self.E
        self.bardelta = (Mbfz % self.p) @ self.delta + self.epsilon

        # print("\nbarD = " + str(self.barD))
        # print("\nbardelta = " + str(self.bardelta))
        # print(Rej(self.barD / self.p, self.D/self.p, self.sigma3, self.e))
        # print(Rej(self.bardelta, self.delta, self.sigma4, self.e))

        return self.A, self.alpha, self.B, self.beta,\
            self.rho, self.barD, self.bardelta


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

        Mbfy = np.diag(bfy)
        Mbfx = np.diag(bfx)
        Mbfz = np.diag(bfz)

        q = self.ck[1]

        # Gather bfA0 and bfAi

        bfA = np.array(
            [bfA0 if i == 0 else bfAi[i-1] for i in range(bfAi.shape[0]+1)])

        # print(commit(A, alpha, self.ck))
        # print(polyeval(bfA, (Mbfx@Mbfy % self.p)))

        lhs1 = commit(A, alpha, self.ck)
        rhs1 = balance(polymodeval(bfA, Mbfx@Mbfy, self.p), q)
        eq1 = (lhs1 == rhs1).all()

        print("Equation 1: " + str(eq1))

        # Gather bfBi and bfBmm1
        bfB = np.array(
            [bfBmm1 if i == self.m else bfBi[i]
             for i in range(bfBi.shape[0]+1)])
        assert(bfB[self.m] == bfBmm1).all()

        eq2 = commit(B, beta, self.ck) \
            == balance(np.sum([(Mbfx**(self.m-i) % self.p) @ bfB[i]
                               for i in range(self.m+1)], axis=0), q)
        eq2 = eq2.all()

        print("Equation 2: " + str(eq2))

        # Calculate rhs of 3rd equation
        rhs = 0
        for i in range(self.m):
            rhs += (Mbfx**(self.m+1) @ Mbfy**i % self.p) @ bfCi[i]
            # The m+1 might be m+1-i

        rhs += polyeval(bfHl, (Mbfx % self.p)) - \
            (Mbfx % self.p)**(self.m + 1) @ bfHl[self.m]

        rhs += bfD

        eq3 = commit((A*B) % self.p, rho, self.ck) == balance(rhs, q)
        eq3 = eq3.all()

        print("Equation 3: " + str(eq3))

        eq4 = commit(barD, bardelta, self.ck) == (Mbfz @ bfD) % self.p + bfE
        eq4 = eq4.all()

        print("Equation 4: " + str(eq4))

        eq5 = barD % self.p == 0
        eq5 = eq5.all()

        print("Equation 5: " + str(eq5))

        eq6 = np.linalg.norm(barD)\
            <= 2*math.sqrt(self.k*self.n)*self.sigma3*self.p

        print("Equation 6: " + str(eq6))

        eq7 = np.linalg.norm(np.hstack((A, alpha, B, beta)))\
            <= 4*math.sqrt(self.k*self.n)*self.sigma1

        print("Equation 7: " + str(eq7))

        eq8 = linalg.norm(rho) <= 2*math.sqrt(self.k*self.n)*self.sigma2

        print("Equation 8: " + str(eq8))

        eq9 = linalg.norm(bardelta)\
            <= 2*math.sqrt(self.k*self.n)*self.sigma4

        print("Equation 9: " + str(eq9))

        return eq1 and eq2 and eq3 and eq4 and eq5\
            and eq6 and eq7 and eq8 and eq9


def productArgument_test():

    lamb = 128
    p = 127
    N = 1000

    np.set_printoptions(threshold=5)

    flatA = np.array([randint(0, p) for _ in range(N)])
    flatB = np.array([randint(0, p) for _ in range(N)])
    flatC = flatA * flatB % p

    # print(flatA)
    # print(flatB)
    # print(flatC)

    argument = ProductArgument(lamb, p, flatA, flatB, flatC)
    print(argument.run())



productArgument_test()

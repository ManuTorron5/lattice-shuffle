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
from util import Rej, gaussian_sample, polyeval, polyeval_no_indep_coeff,\
    matpolymul, balance, polymodeval, polymodeval_no_indep_coeff
# from siszkp import Proof
from commitmentScheme import generateCommitmentKey, commit
from commitAndProve import commit_and_prove, commit_and_prove_all


class ProductArgument(object):

    # Parameters: security level lambda, prime p and N triples ai, bi, ci
    # such that ai*bi = ci (flatA = [a1, a2, ..., aN])

    lamb = None
    p = None
    flatA = None
    flatB = None
    flatC = None

    def __init__(self, lamb, p, flatA, flatB, flatC):
        self.lamb = lamb
        self.p = p
        self.flatA = flatA
        self.flatB = flatB
        self.flatC = flatC

    def run(self):
        assert self.flatA.size == self.flatB.size == self.flatC.size
        N = self.flatA.size

        # Generates the commitment scheme for N values in Zp with security
        # lambda. This defines n, k, m, q ... (?)
        ck = generateCommitmentKey(self.lamb, self.p, N)

        m = ck[5]
        k = ck[4]
        n = ck[3]

        # print(ck)

        # Fill the product triples with zeros so the size matches
        # with the shapes of the new matrices
        self.flatA = np.hstack((self.flatA, np.zeros(m*k*n - N, dtype=int)))
        self.flatB = np.hstack((self.flatB, np.zeros(m*k*n - N, dtype=int)))
        self.flatC = np.hstack((self.flatC, np.zeros(m*k*n - N, dtype=int)))

        Ai = np.resize(self.flatA, (m, k, n))
        Bi = np.resize(self.flatB, (m, k, n))
        Ci = np.resize(self.flatC, (m, k, n))
        
        # Resizing the matrices might be a prover task

        # Create prover
        prover = ProductArgumentProver(self.lamb, self.p, Ai, Bi, Ci, ck)

        # Create verifier
        verifier = ProductArgumentVerifier(self.lamb, self.p, ck)

        # Start protocol
        start_time = time.time()
        proof_size = 0

        bfA0, bfAi, bfBi, bfBmm1, bfCi, commit_witness_communication = prover.commit_witness()

        proof_size += bfA0.size
        proof_size += bfAi.size
        proof_size += bfBi.size
        proof_size += bfBmm1.size
        proof_size += bfCi.size
        proof_size += commit_witness_communication
        
        
        bfy = verifier.generate_challenge()

        print("The verifier sends the challenge y = " + str(bfy))

        bfHl, commit_polynomials_communication = prover.compute_and_commit_polynomials(bfy)

        proof_size += bfHl.size + commit_polynomials_communication

        bfx = verifier.generate_challenge()

        print("The verifier sends the challenge x = " + str(bfx))

        prover.evaluate_polynomials(bfy, bfx)

        bfD, bfE, commit_modulus_communication = prover.modulus_correction(bfy, bfx)

        proof_size += bfD.size
        proof_size += bfE.size
        proof_size += commit_modulus_communication

        bfz = verifier.generate_challenge()

        print("The verifier sends the challenge z = " + str(bfz))

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


class ProductArgumentProver(object):
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

    def __init__(self, lamb, p, Ai, Bi, Ci, ck):

        assert Ai.shape == Bi.shape == Ci.shape
        
        self.lamb = lamb

        self.Ai = Ai
        self.Bi = Bi
        self.Ci = Ci

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
        self.sigma2 = 72*math.sqrt(2*self.k*self.n)*self.k*self.m*self.p
        self.sigma3 = 24*math.sqrt(2*self.k*self.n)*self.k*self.p\
            * (1+6*self.k*self.m*self.p)
        self.sigma4 = 24*math.sqrt(2)*self.k**2*self.p*self.n*self.sigma2

        # print("\nsigma1 = " + str(self.sigma1))
        # print("\nsigma2 = " + str(self.sigma2))
        # print("\nsigma3 = " + str(self.sigma3))
        # print("\nsigma4 = " + str(self.sigma4))

    def commit_witness(self):
        self.Aizeros = np.array(
            [np.vstack((self.Ai[i], np.zeros_like(self.Ai[i], dtype=int)))
             for i in range(self.m)])
        self.Bizeros = np.array(
            [np.vstack((self.Bi[i], np.zeros_like(self.Bi[i], dtype=int)))
             for i in range(self.m)])
        self.CiCpi = (self.Aizeros * self.Bizeros) % self.p
        # This should be done in a suitable basis of GF(p^{2k}) (???)
        
        assert(self.CiCpi[:,0:self.k,:] == self.Ci).all()

        self.A0 = gaussian_sample(self.sigma1, (2*self.k, self.n))
        self.Bmm1 = gaussian_sample(self.sigma1, (2*self.k, self.n))

        self.alphai = np.random.randint(0, self.p,
                                        [self.m, self.k, self.nprime])
        self.betai = np.random.randint(0, self.p,
                                       [self.m, self.k, self.nprime])
        self.gammai = np.random.randint(0, self.p,
                                        [self.m, 2*self.k, self.nprime])

        self.alphaizeros = np.array(
            [np.vstack((self.alphai[i],
                        np.zeros_like(self.alphai[i], dtype=int)))
             for i in range(self.m)])
        self.betaizeros = np.array(
            [np.vstack((self.betai[i],
                        np.zeros_like(self.betai[i], dtype=int)))
             for i in range(self.m)])

        self.alpha0 = gaussian_sample(self.sigma1, (2*self.k, self.nprime))
        self.betamm1 = gaussian_sample(self.sigma1, (2*self.k, self.nprime))
        
        # Commit to matrices A, B, C

        # self.bfAi = np.array(
        #     [commit(self.Aizeros[i], self.alphaizeros[i], self.ck)
        #      for i in range(self.m)])
        # self.bfBi = np.array(
        #     [commit(self.Bizeros[i], self.betaizeros[i], self.ck)
        #      for i in range(self.m)])
        # self.bfCi = np.array(
        #     [commit(self.CiCpi[i], self.gammai[i], self.ck)
        #      for i in range(self.m)])
        
        self.bfAi, verA, proof_size_A =\
            commit_and_prove_all(self.Aizeros, self.alphaizeros, self.ck, self.lamb)
        self.bfBi, verB, proof_size_B =\
            commit_and_prove_all(self.Bizeros, self.betaizeros, self.ck, self.lamb)
        self.bfCi, verC, proof_size_C =\
            commit_and_prove_all(self.CiCpi, self.gammai, self.ck, self.lamb)
            
        commit_witness_communication = proof_size_A + proof_size_B + proof_size_C
        
        print("Running ZKP for commitments of A:", verA)
        print("Running ZKP for commitments of B:", verB)
        print("Running ZKP for commitments of C:", verC)

        # print("\nbfAi = " + str(self.bfAi))
        # print("\nbfBi = " + str(self.bfBi))
        # print("\nbfCi = " + str(self.bfCi))

        self.bfA0 = commit(self.A0, self.alpha0, self.ck)
        self.bfBmm1 = commit(self.Bmm1, self.betamm1, self.ck)

        # print("\nbfA0 = " + str(self.bfA0))
        # print("\nbfBmm1 = " + str(self.bfBmm1))

        return self.bfA0, self.bfAi, self.bfBi, self.bfBmm1, self.bfCi,\
            commit_witness_communication

    def compute_and_commit_polynomials(self, bfy):
        Mbfy = np.diag(bfy)
        # Matrix that emulates left product by bfy in GF(p^{2k}) (???)

        # Compute polynomial A(X) = A0 + Aizeros[0]*y*X +
        # + ... + Aizeros[m-1]*y^m*X^
        self.AX = np.array(
            [self.A0 if i == 0 else (Mbfy**i % self.p) @ self.Aizeros[i-1]
             for i in range(self.m + 1)])


        # Compute polynomial B(X) = B_(m+1) + Bizeros[m-1]*X + Bizeros[m-2]*X²
        # + ... + Bizeros[0]*X^m
        self.BX = np.append([self.Bmm1], np.flipud(self.Bizeros), axis=0)

        # Calculate value of C = CiCpi[0]*y + CiCpi[1]*y² + CiCpi[2]*y³
        # + ... + CiCpi[m-1]*y^m
        self.C = polyeval_no_indep_coeff(self.CiCpi, Mbfy) % self.p
        
        self.Hl = matpolymul(self.AX, self.BX) % self.p
        assert(self.Hl.shape[0] == 2*self.m+1)
        assert(self.Hl[self.m+1] == self.C).all(), "H[m+1] != C"
        

        # Commit to polynomials
        self.etal = np.random.randint(0, self.p,
                                      [2*self.m+1, 2*self.k, self.nprime])

        # self.bfHl = np.array(
        #     [commit(self.Hl[ell], self.etal[ell], self.ck)
        #      for ell in range(2*self.m+1)])
        
        self.bfHl, verH, proof_size_H = commit_and_prove_all(self.Hl, self.etal,
                                                      self.ck, self.lamb)
        
        print("Running ZKP for commitments of H:", verH)
         

        sendbfHl = self.bfHl
        sendbfHl[self.m+1] = np.zeros_like(self.bfHl[0], dtype=int)
        # Send bfHl except m+1 term

        return sendbfHl, proof_size_H

    def evaluate_polynomials(self, bfx, bfy):
        Mbfx = np.diag(bfx)
        Mbfy = np.diag(bfy)
 
        self.A = self.A0
        self.A += polymodeval_no_indep_coeff(self.Aizeros, Mbfx@Mbfy, self.p)
        self.alpha = self.alpha0
        self.alpha += polymodeval_no_indep_coeff(self.alphaizeros, Mbfx@Mbfy, self.p)
        self.B = self.Bmm1
        self.B += polymodeval_no_indep_coeff(np.flipud(self.Bizeros), Mbfx, self.p)
        self.beta = self.betamm1
        self.beta += polymodeval_no_indep_coeff(np.flipud(self.betaizeros), Mbfx, self.p)

        # self.A %= self.p
        # self.alpha %= self.p
        # self.B %= self.p
        # self.beta %= self.p
        # print("\nA = " + str(self.A))
        # print("\nB = " + str(self.B))

        self.Ax = polymodeval(self.AX, Mbfx, self.p)
        self.Bx = polymodeval(self.BX, Mbfx, self.p)

        # print("\nA(x) = " + str(self.Ax))
        # print("\nB(x) = " + str(self.Bx))

        # Check: A(x) = A mod p, B(x) = B mod p
        # assert (self.Ax % self.p == self.A % self.p).all()
        # assert (self.Bx % self.p == self.B % self.p).all()

    def modulus_correction(self, bfy, bfx):
        Mbfy = np.diag(bfy)
        Mbfx = np.diag(bfx)
        
        
        # Calculate D 
        # I think there is a mistake in BBC+18 pg 27: the 2nd summand of D
        # should be multiplied by (M_x)^(m+1) (see page 41)
        # This is D as in the paper:
        # self.D = (self.A * self.B) % self.p\
        #     - polymodeval_no_indep_coeff(self.CiCpi, Mbfy, self.p)\
        #         - polymodeval(self.Hl, Mbfx, self.p)\
        #             + ((Mbfx**(self.m + 1)) % self.p) @ self.Hl[self.m+1]  # l != m+1
        
        # And this is how I think D should be:
       

        self.D = (self.A * self.B) % self.p\
            - (Mbfx**(self.m + 1) % self.p) @ polymodeval_no_indep_coeff(self.CiCpi, Mbfy, self.p)\
                - polymodeval(self.Hl, Mbfx, self.p)\
                    + (Mbfx**(self.m + 1) % self.p) @ self.Hl[self.m+1]  # l != m+1
        
        #assert(self.D % self.p == 0).all()
    
        self.delta = gaussian_sample(self.sigma2, (2*self.k, self.nprime))
        self.bfD, verD, proof_size_D\
           = commit_and_prove(self.D, self.delta, self.ck, self.lamb)
           
        self.E = self.p * gaussian_sample(self.sigma3, (2*self.k, self.n))
        self.epsilon = gaussian_sample(self.sigma4, (2*self.k, self.nprime))
        self.bfE, verE, proof_size_E\
            = commit_and_prove(self.E, self.epsilon, self.ck, self.lamb)

        print("Running ZKP for commitments of D:", verD)
        print("Running ZKP for commitments of E:", verE)
        commit_modulus_communication = proof_size_D + proof_size_E

        assert (self.E % self.p == 0).all

        return self.bfD, self.bfE, commit_modulus_communication

    def rejections(self, bfy, bfx, bfz):
        Mbfy = np.diag(bfy)
        Mbfx = np.diag(bfx)
        Mbfz = np.diag(bfz)

        M1 = np.hstack((self.A, self.alpha, self.B, self.beta))
        # print("\nAalphaBbeta = " + str(M1))
        M2 = np.hstack((self.A0, self.alpha0, self.Bmm1, self.betamm1))
        # print("\nA0alpha0B0beta0 = " + str(M2))
        # self.e = 2
        # print(Rej(M1, M1 - M2, self.sigma1, self.e)) # What is e??
        # Pass all rejections with probability 1/e⁴

        # Calculate rho
        # First summand
        self.rho = 0
        for i in range(self.m):
            self.rho += ((Mbfx**(self.m + 1) @ Mbfy**(i+1)) % self.p) @ self.gammai[i]
            #  The m+1 might be m+1-i

        # Second summand
        self.rho += polymodeval(self.etal, Mbfx, self.p) -\
            (Mbfx**(self.m + 1) % self.p) @ self.etal[self.m+1]
            
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


class ProductArgumentVerifier(object):

    def __init__(self, lamb, p, ck):

        self.lamb = lamb
        self.p = p
        
        self.ck = ck

        self.q = ck[1]
        self.n = ck[3]
        self.k = ck[4]
        self.m = ck[5]

        self.sigma1 = 48*math.sqrt(self.k*self.n)*self.k*self.m*self.p**2
        self.sigma2 = 72*math.sqrt(2*self.k*self.n)*self.k*self.m*self.p
        self.sigma3 = 24*math.sqrt(2*self.k*self.n)*self.k*self.p \
            * (1+6*self.k*self.m*self.p)
        self.sigma4 = 24*math.sqrt(2)*self.k**2*self.p*self.n*self.sigma2

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
        bfA = np.append([bfA0], bfAi, axis=0)

        eq1 = (commit(A, alpha, self.ck)
               == balance(polymodeval(bfA, Mbfx@Mbfy, self.p), q)).all()

        print("Equation 1: " + str(eq1))

        # Gather bfBi and bfBmm1
        bfB = np.append(bfBi, [bfBmm1], axis=0)
        assert(bfB[self.m] == bfBmm1).all()
        
        eq2 = commit(B, beta, self.ck) \
            == balance(polymodeval(np.flipud(bfB), Mbfx, self.p), q)
        eq2 = eq2.all()

        print("Equation 2: " + str(eq2))

        # Calculate rhs of 3rd equation
        rhs = 0
        for i in range(self.m):
            rhs += ((Mbfx**(self.m+1) @ Mbfy**(i+1)) % self.p) @ bfCi[i]

        rhs += polymodeval(bfHl, Mbfx, self.p)  # The (m+1) term of bfHl is zero

        rhs += bfD

        eq3 = commit((A*B), rho, self.ck) == balance(rhs, q)
        eq3 = eq3.all()

        print("Equation 3: " + str(eq3))

        eq4 = commit(barD, bardelta, self.ck) == balance(((Mbfz % self.p)@ bfD) + bfE, q)
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
    p = 4099
    N = 3000

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

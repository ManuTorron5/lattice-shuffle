#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:50:12 2020

@author: Manuel Sánchez Torrón
"""

import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from commitmentScheme import generateCommitmentKey, commit
from siszkp import Proof


"""
Commits and proves to one message in Z_q^(kxn)
We assume the message is already shaped
"""


def commit_and_prove(msg, randomness, ck, lamb):

    committed_message = commit(msg, randomness, ck)

    A = np.hstack((ck[6], ck[7]))
    S = np.hstack((randomness, msg)).T
    T = committed_message.T

    # print(A.shape, S.shape, T.shape)

    # ZKP to convince a verifier that a prover knows S such that A@S = T % q

    proof = Proof(lamb, ck[1], A, S, T)
    bit, size = proof.run()
    # proof.print_stats()
    return committed_message, bit, size


"""
Commits and proves to m messages, each in Z_q^(kxn)
we assume the message is already shaped
"""


def commit_and_prove_all(msg, randomness, ck, lamb):

    m = msg.shape[0]
    # msg = np.hstack((msg, np.zeros(n*k*m - msg.size, dtype=int)))
    # msg = np.resize(msg, (m, k, n))
    # Initialize values with first commitment to get correct shape for com_msg
    com_msg, verification, total_size\
        = commit_and_prove(msg[0], randomness[0], ck, lamb)
    com_msg = [com_msg]

    # Commit and prove the rest of the messages
    for i in range(1, m):

        com_msg_i, bit, size = commit_and_prove(msg[i], randomness[i], ck, lamb)

        com_msg = np.append(com_msg, [com_msg_i], axis=0)
        verification = verification and bit
        total_size += size

    # print("Verification: " +  str(verification))
    # print("Number of elements of communication: " + str(total_size))

    return np.array(com_msg), verification, total_size


def commit_and_prove_test():
    N = 100000  # Number of elements
    lamb = 128
    p = 4099

    msg = np.random.randint((-p+1)//2, (p-1)//2+1, size=N)
    ck = generateCommitmentKey(lamb, p, N)

    nprime = int(2*ck[2]*math.log(ck[1], ck[0]))

    randomness = np.array([randint((-p+1)//2, (p-1)//2+1)
                           for _ in range(ck[5]*ck[4]*nprime)])\
        .reshape(ck[5], ck[4], nprime)

    n, k, m = ck[3], ck[4], ck[5]

    msg = np.resize(msg, (m, k, n))

    print(commit_and_prove_all(msg, randomness, ck, lamb)[1:])
    # [1:] so it does not show the commitments


#commit_and_prove_test()

"""
Test to check sublinearity of proofs for commitment openings
Commits and proves to several messages and plots the number of elements of
communication of each proof
"""


def sublinearity_test():
    lamb = 128
    p = 4099
    msg_sizes = [i for i in range(250, 10001, 250)]
    # msg_sizes = [1000, 2000, 3000, 5000, 7000, 10000, 15000]
    com_sizes = []
    for i in range(len(msg_sizes)):
        # Create random message
        msg = np.random.randint((-p+1)//2, (p-1)//2+1, size=msg_sizes[i])

        # Generate commitment key
        ck = generateCommitmentKey(lamb, p, msg.size)
        
        n, k, m = ck[3], ck[4], ck[5]

        msg = np.resize(msg, (m, k, n))
        
        # Generate randomness
        nprime = int(2*ck[2]*math.log(ck[1], ck[0]))
        randomness = np.array([randint((-p+1)//2, (p-1)//2+1)
                               for _ in range(ck[5]*ck[4]*nprime)])\
            .reshape(ck[5], ck[4], nprime)

        # Run the proof
        _, bit, communication = commit_and_prove_all(msg, randomness, ck, lamb)

        # Store communication size
        print("Proof", i+1, "of", len(msg_sizes), ":", bit)
        com_sizes.append(communication)

    # Print sizes
    plt.plot(msg_sizes, com_sizes, 'bo', msg_sizes, com_sizes, 'b')
    
    plt.show()
    for i in range(len(msg_sizes)-1):
        print((com_sizes[i+1] - com_sizes[i])/(msg_sizes[i+1]-msg_sizes[i]))


sublinearity_test()

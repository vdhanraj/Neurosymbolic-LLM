from nengo.dists import UniformHypersphere
import numpy as np
import json
import torch
import random
import os
import pandas as pd

import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def __init__(SP_dim=4096, max_digits=15):

    rng=np.random.default_rng()

    SP_dim = 4096

    max_digits = 10 # maximum representable number is 10**max_digits

    SP_x  = 0
    SP_n1 = 1
    SP_n2 = 2

    domain_size = 3 + max_digits

    def make_unitary(v):
        """
        Makes input unitary (Fourier components have magnitude of 1)
        """
        fv = np.fft.fft(v, axis=1)
        fv = fv/np.sqrt(fv.real**2 + fv.imag**2)
        return np.fft.ifft(fv, axis=1).real  

    def invert(a, dim):
        """
        Inverts input under binding
        """
        a = np.atleast_2d(a)
        return a[:,-np.arange(dim)]

    def identity(dim):
        """
        Returns
        -------
        np.array
            dim-D identity vector under binding
        """
        s = np.zeros((1, dim))
        s[0][0] = 1
        return s

    def null(dim):
        """
        Returns
        -------
        np.array
            dim-D null vector under binding
        """
        s = np.zeros((1, dim))
        return s

    def bind(a,b):
        """
        Binds togther input

        Parameters
        ----------
        a : np.array
            A vector with shape (n_samples x ssp_dim) 

        b : np.array
            A vector with shape (n_samples x ssp_dim) 

        Returns
        -------
        np.array
            A vector with shape (n_samples x ssp_dim). Row i is a[i,:] binded with b[i,:]

        """
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real


    np.random.seed = 4
    vectors = make_unitary(UniformHypersphere(surface=True).sample(domain_size,SP_dim))
    for j in range(domain_size):
        q = vectors[j,:]/np.linalg.norm(vectors[j,:])
        for k in range(j+1,domain_size):
            vectors[k,:] = vectors[k,:] - (q.T @ vectors[k,:]) * q

    inverse_vectors = invert(vectors, SP_dim)

    digits = {"SP_" + str(10**(i-3)): i for i in range(3, domain_size)}

    vocabulary = {
        "SP_x":    vectors[[SP_x]],
        "SP_n1":   vectors[[SP_n1]],
        "SP_n2":   vectors[[SP_n2]]}

    for d in digits:
        vocabulary[d] = vectors[[digits[d]]]


    vocabulary_inverse = {
        "SP_x":    inverse_vectors[[SP_x]],
        "SP_n1":   inverse_vectors[[SP_n1]],
        "SP_n2":   inverse_vectors[[SP_n2]]}

    for d in digits:
        vocabulary_inverse[d] = inverse_vectors[[digits[d]]]

    num_SP = identity(SP_dim).reshape(1, -1)
    numbers = {}
    numbers["0"] = num_SP
    for i in range(1, 10):
        num_SP = bind(num_SP, vectors[SP_x])
        vocabulary["SP_number_" + str(i)] = num_SP
        numbers[str(i)] = num_SP

    def generate_SP(num1, num2):
        
        nums1_coefs = [int(i) for i in list(str(num1))][::-1]
        nums2_coefs = [int(i) for i in list(str(num2))][::-1]

        total_SP1 = np.zeros((1, SP_dim))
        for digit in range(len(nums1_coefs)):
            num_SP = identity(SP_dim)
            for i in range(nums1_coefs[digit]):
                num_SP = bind(num_SP, vectors[SP_x])
            #print("SP_" + str(10**digit), nums1_coefs[digit])
            num_SP = bind(num_SP, vocabulary["SP_" + str(10**digit)])

            total_SP1 += num_SP

        total_SP1 = bind(total_SP1, vectors[SP_n1])

        total_SP2 = np.zeros((1, SP_dim))
        for digit in range(len(nums2_coefs)):
            num_SP = identity(SP_dim)
            for i in range(nums2_coefs[digit]):
                num_SP = bind(num_SP, vectors[SP_x])

            #print("SP_" + str(10**digit), nums2_coefs[digit])
            num_SP = bind(num_SP, vocabulary["SP_" + str(10**digit)])

            total_SP2 += num_SP

        total_SP2 = bind(total_SP2, vectors[SP_n2])

        final_SP = total_SP1 + total_SP2
        
        return final_SP

    def decode_SP(SP, SP_n=None, similarity_threshold=0.5):
        if SP_n:
            n = bind(SP, inverse_vectors[SP_n])
        else:
            n = SP

        coefs = {}

        query = bind(n, inverse_vectors[list(digits.values())])

        digit_values = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).argmax(axis=0)
        digit_scores = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).max(axis=0)
        digit_values[digit_scores < similarity_threshold] = -1
        
        for n, d in enumerate(digits):
            if digit_values[n] != -1:
                coefs[d] = digit_values[n]
        
        if len(coefs):
            return int("".join([str(i) for i in list(coefs.values())][::-1]))
        else:
            return 0

    def single_digit_addition(n1, n2, n3):
        # TODO: If desired, we can make this fully in SP operations, instead of doing n1 + n2 + n3, modulo, and integer division
        ones_SP = identity(SP_dim)
        tens_digit = (n1 + n2 + n3) // 10
        ones_digit = (n1 + n2 + n3) %  10
        for i in range(ones_digit):
            ones_SP = bind(ones_SP, vectors[SP_x])
        tens_SP = identity(SP_dim)
        for i in range(tens_digit):
            tens_SP = bind(tens_SP, vectors[SP_x])
        remainder = bind(tens_SP, vocabulary["SP_1"])
        #resultant_SP = bind(ones_SP, vocabulary["SP_1"]) + bind(tens_SP, vocabulary["SP_10"])
        #resultant_SP = bind(ones_SP, vocabulary["SP_1"])
        return ones_SP, remainder

    def multiply_SP(SP, similarity_threshold=0.5):
        n1 = bind(SP, vocabulary_inverse["SP_n1"])
        n2 = bind(SP, vocabulary_inverse["SP_n2"])

        coefs1 = {}
        coefs2 = {}

        query = bind(n1, inverse_vectors[list(digits.values())])

        digit_values = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).argmax(axis=0)
        digit_scores = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).max(axis=0)
        digit_values[digit_scores < similarity_threshold] = -1

        for n, d in enumerate(digits):
            if digit_values[n] != -1:
                coefs1[d] = digit_values[n]

        query = bind(n2, inverse_vectors[list(digits.values())])

        digit_values = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).argmax(axis=0)
        digit_scores = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).max(axis=0)
        digit_values[digit_scores < similarity_threshold] = -1

        for n, d in enumerate(digits):
            if digit_values[n] != -1:
                coefs2[d] = digit_values[n]

        print(coefs1, coefs2)

        # TODO: Complete multiplication process

    def add_SP(SP, similarity_threshold=0.5):
        n1 = bind(SP, vocabulary_inverse["SP_n1"])
        n2 = bind(SP, vocabulary_inverse["SP_n2"])

        coefs1 = {}
        coefs2 = {}

        query = bind(n1, inverse_vectors[list(digits.values())])

        digit_values = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).argmax(axis=0)
        digit_scores = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).max(axis=0)
        digit_values[digit_scores < similarity_threshold] = -1

        for n, d in enumerate(digits):
            if digit_values[n] != -1:
                coefs1[d] = digit_values[n]

        query = bind(n2, inverse_vectors[list(digits.values())])

        digit_values = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).argmax(axis=0)
        digit_scores = np.dot(np.array(list(numbers.values())).reshape(-1, SP_dim), query.T).max(axis=0)
        digit_values[digit_scores < similarity_threshold] = -1

        for n, d in enumerate(digits):
            if digit_values[n] != -1:
                coefs2[d] = digit_values[n]

        max_digit = max(len(coefs1), len(coefs2))

        sum_SP = null(SP_dim)
        r      = 0 # remainder
        for n, d in enumerate(digits):
            n1, n2 = 0, 0
            if d in coefs1:
                n1 = coefs1[d]
            if d in coefs2:
                n2 = coefs2[d]
            if n1 == n2 == r == 0:
                break
            #print(n1, n2, r)
            value_SP, remainder_SP = single_digit_addition(n1, n2, r)

            r = decode_SP(remainder_SP)
            sum_SP += bind(value_SP, vocabulary[d])
        return sum_SP


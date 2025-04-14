#from nengo.dists import UniformHypersphere
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

from llama.encoder_decoder_networks import *

def make_unitary(v):
    """
    Makes input unitary (Fourier components have magnitude of 1)
    """
    fv = torch.fft.fft(v.double(), axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    return torch.fft.ifft(fv, axis=1).real.type_as(v)

def make_tensor_unitary(v):
    """
    Makes input tensor unitary (Fourier components have magnitude of 1)
    """
    fv = torch.fft.fft(v, axis=1)
    fv = fv/torch.sqrt(fv.real**2 + fv.imag**2)
    return torch.fft.ifft(fv, axis=1).real

def invert(a, dim):
    """
    Inverts input under binding
    """
    return a[:,-torch.arange(dim, device=a.device)]

def identity(dim):
    """
    Returns
    -------
    torch.tensor
        dim-D identity vector under binding
    """
    s = torch.zeros((1, dim), dtype=torch.float32)
    s[0][0] = 1
    return s

def null(shape):
    """
    Returns
    -------
    torch.tensor
        dim-D null vector under binding
    """
    s = torch.zeros(shape, dtype=torch.float32)
    return s

def bind(a, b):
    """
    Binds together input using torch tensors.

    Parameters
    ----------
    a : torch.Tensor
        A tensor with shape (n_samples x vsa_dim)

    b : torch.Tensor
        A tensor with shape (n_samples x vsa_dim)

    Returns
    -------
    torch.Tensor
        A tensor with shape (n_samples x vsa_dim). Row i is a[i,:] bound with b[i,:]
    """
    ## Ensure the inputs are at least 2D (this behavior mimics np.atleast_2d)
    a = a.unsqueeze(0) if a.dim() == 1 else a
    b = b.unsqueeze(0) if b.dim() == 1 else b
    
    # Compute the FFT of both inputs along the last dimension
    fft_a = torch.fft.fft(a, dim=1)
    fft_b = torch.fft.fft(b, dim=1)

    # Multiply the FFTs element-wise
    fft_product = fft_a * fft_b

    # Perform the inverse FFT and take the real part of the result
    bound = torch.fft.ifft(fft_product, dim=1).real

    return bound

def SampleUniformHypersphere(surface, n, d, rng=torch, min_magnitude=1):
    if d is None or d < 1:
        raise ValueError("Dimensions must be a positive integer", 'd')

    # Generate n x d random samples from a normal distribution
    samples = rng.randn(n, d)
    samples = samples / samples.norm(dim=1, keepdim=True)  # Normalize along the last dimension

    if surface:
        return samples

    # Generate magnitudes for vectors from a uniform distribution.
    # The (1 / d) exponent ensures that samples are uniformly distributed
    # in n-space and not all bunched up at the centre of the sphere.
    samples = samples * (rng.rand(n, 1) * (1 - min_magnitude**d) + min_magnitude**d) ** (1. / d)
    return samples

class SymbolicEngine():
    def __init__(self, VSA_dim=4096, max_digits=15, possible_problems=["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"],
                 curr_dir=".", similarity_threshold=0.5, seed=4):
        torch.manual_seed(seed)

        self.VSA_dim = VSA_dim

        self.similarity_threshold = similarity_threshold

        self.max_digits = max_digits # maximum representable number is 10**max_digits

        self.VSA_x  = 0
        self.VSA_n1 = 1
        self.VSA_n2 = 2

        self.possible_problems = possible_problems

        self.VSA_digit = 3 + len(self.possible_problems)

        self.domain_size = 4 + len(self.possible_problems)

        if not os.path.exists(f"{curr_dir}/VSA_library"):
            os.mkdir(f"{curr_dir}/VSA_library")
            print("Created VSA_library directory")

        if os.path.exists(f"{curr_dir}/VSA_library/VSA_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt"):
            self.vectors         = torch.load(f"{curr_dir}/VSA_library/VSA_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt", weights_only=True)
            self.inverse_vectors = torch.load(f"{curr_dir}/VSA_library/VSA_inverse_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt", weights_only=True)
            print("Using existing VSAs:\n", 
                f"{curr_dir}/VSA_library/VSA_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt\n",
                f"{curr_dir}/VSA_library/VSA_inverse_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt"
                )
        else:
            torch.random.seed = 4
            self.vectors = make_unitary(SampleUniformHypersphere(surface=True, n=self.domain_size, d=VSA_dim)).to(torch.float32)
            for j in range(self.domain_size):
                q = self.vectors[j,:]/torch.linalg.norm(self.vectors[j,:])
                for k in range(j+1,self.domain_size):
                    self.vectors[k,:] = self.vectors[k,:] - (q @ self.vectors[k,:]) * q
            self.vectors = make_tensor_unitary(self.vectors)
            self.inverse_vectors = invert(self.vectors, VSA_dim)
            torch.save(self.vectors, f"{curr_dir}/VSA_library/VSA_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt")
            torch.save(self.inverse_vectors, f"{curr_dir}/VSA_library/VSA_inverse_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt")
            print("Saved VSAs:\n", 
                f"{curr_dir}/VSA_library/VSA_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt\n",
                f"{curr_dir}/VSA_library/VSA_inverse_vector_library_VSAdim_{VSA_dim}_domainSize_{self.domain_size}.pt"
                )

        self.digits = {"VSA_" + str(10**(i-3-len(self.possible_problems))): i for i in range(3+len(self.possible_problems), 3+len(self.possible_problems)+max_digits)}

        self.vocabulary = {
            "VSA_x":    self.vectors[[self.VSA_x]],
            "VSA_n1":   self.vectors[[self.VSA_n1]],
            "VSA_n2":   self.vectors[[self.VSA_n2]]}

        for n, pt in enumerate(self.possible_problems):
            self.vocabulary[pt] = self.vectors[[len(self.vocabulary)]]


        new_digit_tensors = []
        for n, d in enumerate(self.digits):
            if n == 0:
                VSA = self.vectors[[self.VSA_digit]]
            else:
                VSA = bind(VSA, VSA)
                new_digit_tensors += [VSA.flatten()]
            self.vocabulary[d] = VSA

        self.vectors = torch.cat((self.vectors, torch.stack(new_digit_tensors)), dim=0)

        self.vocabulary_inverse = {
            "VSA_x":    self.inverse_vectors[[self.VSA_x]],
            "VSA_n1":   self.inverse_vectors[[self.VSA_n1]],
            "VSA_n2":   self.inverse_vectors[[self.VSA_n2]]}

        for n, pt in enumerate(self.possible_problems):
            self.vocabulary_inverse[pt] = self.inverse_vectors[[len(self.vocabulary_inverse)]]

        new_digit_tensors = []
        for n, d in enumerate(self.digits):
            if n == 0:
                VSA = self.inverse_vectors[[self.VSA_digit]]
            else:
                VSA = bind(VSA, VSA)
                new_digit_tensors += [VSA.flatten()]
            self.vocabulary_inverse[d] = VSA

        self.inverse_vectors = torch.cat((self.inverse_vectors, torch.stack(new_digit_tensors)), dim=0)

        num_VSA = identity(self.VSA_dim).reshape(1, -1)
        self.numbers = {}
        self.numbers["0"] = num_VSA
        self.vocabulary["VSA_number_" + str(0)] = num_VSA
        for i in range(1, 10):
            num_VSA = bind(num_VSA, self.vectors[self.VSA_x])
            self.vocabulary["VSA_number_" + str(i)] = num_VSA
            self.numbers[str(i)] = num_VSA


    def generate_VSA(self, num1, num2, problem_type="addition", single_number_generation=False): # num1, num2 are integers
        nums1_coefs = [int(i) for i in list(str(num1))][::-1]

        total_VSA1 = torch.zeros((1, self.VSA_dim)).float()
        for digit in range(len(nums1_coefs)):
            num_VSA = identity(self.VSA_dim).float()
            for i in range(nums1_coefs[digit]):
                num_VSA = bind(num_VSA, self.vectors[self.VSA_x])
            #print("VSA_" + str(10**digit), nums1_coefs[digit])
            num_VSA = bind(num_VSA, self.vocabulary["VSA_" + str(10**digit)])

            total_VSA1 += num_VSA

        if not single_number_generation:
            total_VSA1 = bind(total_VSA1, self.vectors[self.VSA_n1])
            nums2_coefs = [int(i) for i in list(str(num2))][::-1]
            total_VSA2 = torch.zeros((1, self.VSA_dim)).float()
            for digit in range(len(nums2_coefs)):
                num_VSA = identity(self.VSA_dim).float()
                for i in range(nums2_coefs[digit]):
                    num_VSA = bind(num_VSA, self.vectors[self.VSA_x])

                #print("VSA_" + str(10**digit), nums2_coefs[digit])
                num_VSA = bind(num_VSA, self.vocabulary["VSA_" + str(10**digit)])

                total_VSA2 += num_VSA

            total_VSA2 = bind(total_VSA2, self.vectors[self.VSA_n2])

            final_VSA = total_VSA1 + total_VSA2 + self.vocabulary[problem_type]

        else:
            final_VSA = total_VSA1

        return final_VSA

    def decode_VSA(self, VSA, VSA_n=None, differentiable=False, similarity_threshold=0.5, T=0.01, exp_scalar=100, k=100):
        if VSA_n:
            n = bind(VSA, self.inverse_vectors[VSA_n])
        else:
            n = VSA

        # Bind the input VSA with each inverse VSA for each digit. Output is num_digits x VSA_dim
        query = bind(n, self.inverse_vectors[list(self.digits.values())])

        # Bind the above query with the VSA for each number (size of this is batch x unique_numbers x VSA_dim)
        # Output is batch x unique_numbers x num_digits (this indicates the vector similarity per digit of each number)
        vs = (torch.stack(list(self.numbers.values())).reshape(-1, len(self.numbers), self.VSA_dim) @ query.T)
        batch_size = vs.shape[0]

        if differentiable:
            digit_values = torch.softmax(vs/T, dim=1)
            digit_scores = 1/exp_scalar*torch.log(torch.exp(vs*exp_scalar).sum(dim=1)) # LogSumExp
            digit_scores = torch.sigmoid(k * (digit_scores - similarity_threshold))
            modified_digit_values = digit_values * digit_scores.unsqueeze(1)
            
            exponents = torch.tensor([10**d for d in range(len(self.digits))], dtype=torch.float32)
            nums = torch.arange(0, 10, dtype=torch.float32)

            # The i loop iterates over the different digits, and the j loop iterates over the different batch elements
            decoded_VSAs = torch.stack([sum([(exponents[i] * torch.dot(nums.double(), modified_digit_values[j,:,i].double()))
                                            for i in range(self.max_digits)])
                                    for j in range(batch_size)]).to(VSA.device)

        else:
            modified_digit_values = []
            for batch_item in range(vs.shape[0]):
                digit_scores = vs[batch_item].max(axis=0).values
                digit_values = vs[batch_item].max(axis=0).indices
                modified_digit_value = []
                for i in range(vs.shape[2]):
                    if digit_scores[i] > similarity_threshold:
                        modified_digit_value += [digit_values[i]]
                    else:
                        modified_digit_value += [0.]
                modified_digit_value = torch.tensor(modified_digit_value)
                modified_digit_values += [modified_digit_value]
            modified_digit_values = torch.stack(modified_digit_values, dim=0).type(torch.float32)

            exponents = torch.tensor([10**d for d in range(len(self.digits))], dtype=torch.float32)
            nums = torch.arange(0, 10, dtype=torch.float32)

            # The i loop iterates over the different digits, and the j loop iterates over the different batch elements
            decoded_VSAs = torch.stack([sum([(exponents[i] * modified_digit_values[j,i])
                                            for i in range(self.max_digits)])
                                        for j in range(batch_size)]).to(VSA.device)

        return decoded_VSAs

    # Make sure similarity_threshold is picked such that the correct number has a score greater than this threshold, and everything 
    #  else is smaller than the threshold
    def query_digit(self, VSA, d, similarity_threshold=0.5, T=0.01, exp_scalar=100, k=100, differentiable=False):
        # Bind the input VSA with each inverse VSA for each digit. Output is num_digits x VSA_dim
        query = bind(VSA, self.inverse_vectors[self.digits[d]])

        # Bind the above query with the VSA for each number (size of this is batch x unique_numbers x VSA_dim)
        # Output is batch x num_digits (this indicates the vector similarity per digit of each number)
        vs = (torch.stack(list(self.numbers.values())).reshape(-1, len(self.numbers), self.VSA_dim) @ query.mT).mT.squeeze(0)

        if differentiable:
            digit_values = torch.softmax(vs/T, dim=1)
            digit_scores = 1/exp_scalar*torch.log(torch.exp(vs*exp_scalar).sum(dim=1)+1) # LogSumExp
            digit_scores = torch.sigmoid(k * (digit_scores - similarity_threshold))
            modified_digit_values = digit_values * digit_scores.unsqueeze(1)

            nums = torch.arange(0, 10, dtype=torch.float32)
            queried_digit = (nums * modified_digit_values).sum(axis=-1)
        else:
            digit_scores = vs.max(axis=1).values
            digit_values = vs.max(axis=1).indices
            queried_digit = (digit_scores > similarity_threshold) * digit_values

        return queried_digit

    def decode_digits(self, VSA, n=0, verbose=False, differentiable=False):
        if n == 0: 
            n == self.VSA_n1
        decoded_values = []
        for d in self.digits.keys():
            dv = (self.query_digit(bind(VSA, self.inverse_vectors[n].cuda()), d, differentiable=differentiable))
            if verbose:
                print(d, '\t', round(dv, 3))
            decoded_values += [dv]
        decoded_values = torch.stack(decoded_values).T 
        return decoded_values

    def digit_error(self, predictions, labels, error_per_digit=False, verbose=0):
        # Expects predictions and labels to be torch tensors (of type int) to be the same length
        if verbose == 2:
            print("Predictions:")
            print(predictions)
            print("Labels:")
            print(labels)

        errors       = torch.zeros(len(predictions), dtype=int)
        digit_errors = torch.zeros(self.max_digits,  dtype=int)
        for i in range(self.max_digits):
            digit_errors[i] = ((predictions % 10 ** (i + 1) // 10 ** i) != (labels % 10 ** (i + 1) // 10 ** i)).sum()
            errors += (predictions % 10 ** (i + 1) // 10 ** i) != (labels % 10 ** (i + 1) // 10 ** i)
        if error_per_digit:
            return errors, digit_errors
        else:
            return errors

    def decode_problem_type(self, VSA, problem_subset=None, normalize_VSA_before_dot=False, verbose=False):
        # Compute the L2 norm along the second dimension (dim=1)
        l2_norm = torch.norm(VSA, p=2, dim=1, keepdim=True)  # Keep dimensions for broadcasting

        # Normalize VSA to have unit length along the second dimension
        VSA_normalized = VSA / l2_norm

        if verbose:
            print("VSA_norm before normalization:", torch.norm(VSA,            p=2, dim=1, keepdim=False))
            print("L2 norms:", l2_norm.squeeze())  # Show the L2 norms for verification
            print("VSA_norm after normalization:",  torch.norm(VSA_normalized, p=2, dim=1, keepdim=False))  # Mean along dim 1

        if normalize_VSA_before_dot:
            VSA_curr = VSA_normalized
        else:
            VSA_curr = VSA

        if not problem_subset:
            problem_subset = self.possible_problems
        problem_type_VSAs = []
        for p in problem_subset:
            problem_type_VSAs += [self.vocabulary[p].flatten()]
        problem_type_VSAs = torch.stack(problem_type_VSAs)
        problem_type_maxima = (problem_type_VSAs @ VSA_curr.T.float()).T.max(axis=1)
        problem_type_labels = [problem_subset[i] for i in problem_type_maxima.indices]
        if verbose:
            print(problem_type_labels)
        return np.array(problem_type_labels), problem_type_maxima.values.cpu().numpy(), (problem_type_VSAs @ VSA_curr.T.float()).T.cpu().numpy()
    
    def fractional_encode(self, x):
        return torch.fft.ifft((torch.fft.fft(self.vocabulary["VSA_x"])**x.view(-1, 1))).real

    def single_digit_addition_fourier(self, n1, n2, n3, sum_terms=500, epsilon=0.25):
        n = n1 + n2 + n3 + epsilon

        # Parameters
        batch_size = n.size(0)
        max_digit = 2 + 1
        sum_terms = sum_terms + 1

        # Prepare indices in a vectorized way
        k_values = torch.arange(1, sum_terms, dtype=torch.float32).view(1, 1, -1)  # Shape (1, 1, sum_terms - 1)
        d_values = torch.arange(max_digit, dtype=torch.float32).view(1, -1, 1)        # Shape (1, max_digit, 1)

        # Reshape n for broadcasting across digits and sum terms
        n = n.view(batch_size, 1, 1)  # Shape (batch, 1, 1)

        # Calculate A in a fully vectorized way
        A = torch.sin(2 * torch.pi * k_values * n / 10**d_values) / k_values  # Shape (batch, max_digit, sum_terms - 1)

        # Summations with the computed A, applied across the batch
        n_ones = 4.5 + (1 / torch.pi) * (A[:, 0, :] - 10 * A[:, 1, :]).sum(dim=1)
        n_tens = torch.relu(4.5 + (1 / torch.pi) * (A[:, 1, :] - 10 * A[:, 2, :]).sum(dim=1))

        return n_ones, n_tens


    # Method to add two numbers represented by a VSA (n1_tag * n1 + n2_tag * n2) in a differentiable way
    def add_VSA(self, VSA, similarity_threshold=0.5):
        #print("VSA", VSA.shape)
        #VSA = VSA / torch.sqrt(torch.sum(VSA ** 2))
        n1 = bind(VSA, self.vocabulary_inverse["VSA_n1"])
        n2 = bind(VSA, self.vocabulary_inverse["VSA_n2"])

        n3 = null(VSA.shape)
        r  = null((VSA.shape[0]))
        for d in self.digits.keys():
            digit_n1 = self.query_digit(n1, d, similarity_threshold=similarity_threshold)
            digit_n2 = self.query_digit(n2, d, similarity_threshold=similarity_threshold)
            #print("digit:", d, "digit1:", digit_n1.item(), "digit2:", digit_n2.item(), "remainder:", r.item(), "sum:", digit_n1.item() + digit_n2.item() + r.item())

            digit_n3, r = self.single_digit_addition_fourier(digit_n1, digit_n2, r)
            #print("post addition digits", digit_n3.item(), r.item())

            n3 = n3 + bind(self.vocabulary[d], self.fractional_encode(digit_n3))
            #print("n3", n3, "\n")
        return n3

    def generate_counter(self, n, batch_size):
        if n == 0:
            counter_VSA = torch.zeros((batch_size, self.VSA_dim))
        else:
            counter_VSA = identity(self.VSA_dim).reshape(1, -1)
            for _ in range(n):
                counter_VSA = bind(counter_VSA, self.vectors[self.VSA_x])

        counter_VSA = counter_VSA.expand(batch_size, -1).to(dtype=torch.bfloat16)
        return counter_VSA

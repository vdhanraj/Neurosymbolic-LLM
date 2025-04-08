import json
import torch
import numpy as np
import random
import os
import pandas as pd
import sys
import random
import math
import argparse

#sys.path.append('/u4/vdhanraj/Neurosymbolic-LLM/llama/llama3/llama')
#sys.path.append('/u4/vdhanraj/Neurosymbolic-LLM/llama/llama3')
sys.path.insert(0, '/u4/vdhanraj/Neurosymbolic-LLM')

#import openai
#import requests
#from requests.adapters import HTTPAdapter
#from requests.packages.urllib3.util.retry import Retry

#import time
#from pathlib import Path
from typing import List

import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.generation import sample_top_p
from llama.EncoderNetworks import Encoder, Decoder, Encoder_Deep, Decoder_Deep
from llama.vsa_engine import *

from typing import List, Optional
import fire

from llama import Dialog, Llama

from IPython.display import Markdown

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import wandb

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go

#from nengo.dists import UniformHypersphere
from pathlib import Path

import datetime

parser = argparse.ArgumentParser(description="Train Encoders and Decoders")

# Define arguments
parser.add_argument("--curr_dir", type=str, required=False, help="Directory of Program", default="~/Neurosymbolic-LLM/Programs")
parser.add_argument("--chpt_dir", type=str, required=False, help="Model Checkpoint Directory", default="~/.llama/checkpoints/Llama3.1-8B-Instruct")
parser.add_argument("--tokenizer_path", type=str, required=False, help="Tokenizer Checkpoint Directory", default="~/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model")
parser.add_argument("--generate_data", type=bool, required=False, help="Whether to generate training data for encoder or to train encoder and decoder", default=False)

args = parser.parse_args()

curr_dir       = str(Path(args.curr_dir).expanduser())
ckpt_dir       = str(Path(args.chpt_dir).expanduser())
tokenizer_path = str(Path(args.tokenizer_path).expanduser())
# If true, load the LLM and generate the data used to train the encoder
# If false, don't put LLM into memory and train encoder instead
generate_data  = bool(args.generate_data)


curr_date = datetime.datetime.now().strftime("%Y%m%d")

wandb.finish() # If there is an active current run, terminate it
if generate_data:
    wandb.init(
        project = "Symbolic LLM - Generate Encoder Input Data",
        name    = f"{curr_date}",
    )
else:
    wandb.init(
        project = "Symbolic LLM - Train Encoders and Decoders",
        name    = f"{curr_date}",
    )

#print("ckpt_dir:", ckpt_dir, "tokenizer_path:", tokenizer_path, "generate_data:", generate_data)

max_seq_len = 10000
max_batch_size = 2
model_parallel_size = 1

top_p = 0.9
temperature = 0
max_gen_len = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(torch.cuda.current_device()))



if generate_data:
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.2"
    os.environ['MASTER_PORT'] = "29502"
    os.environ['LOCAL_RANK']  = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    self = generator


if generate_data:
    def episode(dialogs, temperature=0.0, top_p=0.9, inference_mode=self.model.forward, skip_weight=0.5, 
                max_decoding_length=100):
        #print(dialogs)
        prompt_tokens = generator.parse_chat(dialogs)

        max_gen_len = self.model.params.max_seq_len - 1
        top_p = top_p
        echo = False

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        transitions = []
        num_tokens = 0
        list_of_probs = []
        h_stacks = []
        for cur_pos in range(min_prompt_len, total_len):
            logits, h_stack, h = inference_mode(tokens[:, prev_pos:cur_pos], prev_pos, skip_weight=skip_weight)
            h_stacks += [h_stack]
            # probs are intentionally being calculated here, so that it contains an extra token (the stop token), to help with loss calculation
            probs = torch.softmax(logits[:, -1] / 1, dim=-1)
            list_of_probs += [probs]
            new_logits = logits
            if temperature > 0:
                probs = torch.softmax(new_logits[:, -1] / temperature, dim=-1)
                #print(logits, logits.shape)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(new_logits[:, -1], dim=-1)
            if num_tokens > max_decoding_length:
                next_token = stop_tokens[0]
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

            #probs = torch.softmax(logits[:, -1] / 1, dim=-1)
            #list_of_probs += [probs]


            num_tokens += 1

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)

        return h_stacks, list_of_probs, out_tokens



max_digits = 10 #15 # maximum representable number is 10**max_digits
SP_dim = 2048

SP_x = 0
SP_n1 = 1
SP_n2 = 2

possible_problems = ["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"]

SP_digit = 3 + len(possible_problems)

domain_size = 4 + len(possible_problems)

if not os.path.exists(f"{curr_dir}/SP_library"):
    os.mkdir(f"{curr_dir}/SP_library")


if os.path.exists(f"{curr_dir}/SP_library/SP_vector_library_SPdim_{SP_dim}_domainSize_{domain_size}.pt"):
    vectors         = torch.load(f"{curr_dir}/SP_library/SP_vector_library_SPdim_{SP_dim}_domainSize_{domain_size}.pt")
    inverse_vectors = torch.load(f"{curr_dir}/SP_library/SP_inverse_vector_library_SPdim_{SP_dim}_domainSize_{domain_size}.pt")
else:
    torch.random.seed = 4
    vectors = torch.tensor(make_unitary(SampleUniformHypersphere(surface=True, n=domain_size, d=SP_dim)), dtype=torch.float32).cuda()
    for j in range(domain_size):
        q = vectors[j,:]/torch.linalg.norm(vectors[j,:])
        for k in range(j+1,domain_size):
            vectors[k,:] = vectors[k,:] - (q.T @ vectors[k,:]) * q
    vectors = make_tensor_unitary(vectors)
    inverse_vectors = invert(vectors, SP_dim).cuda()
    torch.save(vectors, f"{curr_dir}/SP_library/SP_vector_library_SPdim_{SP_dim}_domainSize_{domain_size}.pt")
    torch.save(inverse_vectors, f"{curr_dir}/SP_library/SP_inverse_vector_library_SPdim_{SP_dim}_domainSize_{domain_size}.pt")

digits = {"SP_" + str(10**(i-3-len(possible_problems))): i for i in range(3+len(possible_problems), 3+len(possible_problems)+max_digits)}

vocabulary = {
    "SP_x":    vectors[[SP_x]],
    "SP_n1":   vectors[[SP_n1]],
    "SP_n2":   vectors[[SP_n2]]}

for n, pt in enumerate(possible_problems):
    vocabulary[pt] = vectors[[len(vocabulary)]]

new_digit_tensors = []
for n, d in enumerate(digits):
    if n == 0:
        SP = vectors[[SP_digit]]
    else:
        SP = bind(SP, SP)
        new_digit_tensors += [SP.flatten()]
    vocabulary[d] = SP

vectors = torch.cat((vectors, torch.stack(new_digit_tensors)), dim=0)

vocabulary_inverse = {
    "SP_x":    inverse_vectors[[SP_x]],
    "SP_n1":   inverse_vectors[[SP_n1]],
    "SP_n2":   inverse_vectors[[SP_n2]]}

for n, pt in enumerate(possible_problems):
    vocabulary_inverse[pt] = inverse_vectors[[len(vocabulary_inverse)]]

new_digit_tensors = []
for n, d in enumerate(digits):
    if n == 0:
        SP = inverse_vectors[[SP_digit]]
    else:
        SP = bind(SP, SP)
        new_digit_tensors += [SP.flatten()]
    vocabulary_inverse[d] = SP

inverse_vectors = torch.cat((inverse_vectors, torch.stack(new_digit_tensors)), dim=0)

num_SP = identity(SP_dim).reshape(1, -1).cuda()
number_SPs = {}
number_SPs["0"] = num_SP
vocabulary["SP_number_" + str(0)] = num_SP
for i in range(1, 10):
    num_SP = bind(num_SP, vectors[SP_x])
    vocabulary["SP_number_" + str(i)] = num_SP
    number_SPs[str(i)] = num_SP

def generate_SP(num1, num2, problem_type="bitwise_xor"):
    nums1_coefs = [int(i) for i in list(str(num1))][::-1]
    nums2_coefs = [int(i) for i in list(str(num2))][::-1]

    total_SP1 = torch.zeros((1, SP_dim), dtype=torch.float32).cuda()
    for digit in range(len(nums1_coefs)):
        num_SP = identity(SP_dim).cuda()
        for i in range(nums1_coefs[digit]):
            num_SP = bind(num_SP, vectors[SP_x])
        #print("SP_" + str(10**digit), nums1_coefs[digit])
        num_SP = bind(num_SP, vocabulary["SP_" + str(10**digit)])

        total_SP1 += num_SP

    total_SP1 = bind(total_SP1, vectors[SP_n1])

    total_SP2 = torch.zeros((1, SP_dim), dtype=torch.float32).cuda()
    for digit in range(len(nums2_coefs)):
        num_SP = identity(SP_dim).cuda()
        for i in range(nums2_coefs[digit]):
            num_SP = bind(num_SP, vectors[SP_x])

        #print("SP_" + str(10**digit), nums2_coefs[digit])
        num_SP = bind(num_SP, vocabulary["SP_" + str(10**digit)])

        total_SP2 += num_SP

    total_SP2 = bind(total_SP2, vectors[SP_n2])

    final_SP = total_SP1 + total_SP2 + vocabulary[problem_type]

    return final_SP

num1, num2 = 3, 2
#num1, num2 = 38673028106, 6834025840
final_SP = generate_SP(num1, num2)

def decode_SP(SP, SP_n=None, similarity_threshold=0.5, T=0.01, exp_scalar=100, k=100):
    if SP_n:
        n = bind(SP, inverse_vectors[SP_n])
    else:
        n = SP

    query = bind(n, inverse_vectors[list(digits.values())])

    vs = (torch.stack(list(number_SPs.values())).reshape(-1, len(number_SPs), SP_dim) @ query.T)

    digit_values = torch.softmax(vs/T, dim=1)
    digit_scores = 1/exp_scalar*torch.log(torch.exp(vs*exp_scalar).sum(dim=1)) # LogSumExp
    digit_scores = torch.sigmoid(k * (digit_scores - similarity_threshold))
    modified_digit_values = digit_values * digit_scores.unsqueeze(1)

    exponents = torch.tensor([10**d for d in range(len(digits))], dtype=torch.float32)
    nums = torch.arange(0, 10, dtype=torch.float32).cuda()

    decoded_SPs = torch.stack([sum([(exponents[i] * torch.dot(nums.double(), modified_digit_values[j,:,i].double()))
                                    for i in range(digit_values.shape[2])])
                               for j in range(digit_values.shape[0])]).to(SP.device)

    return decoded_SPs

decode_SP(final_SP, SP_n1).item(), decode_SP(final_SP, SP_n2).item()

def single_digit_addition_fourier(n1, n2, n3, sum_terms=1000, epsilon=0.25):
    n = (n1 + n2 + n3 + epsilon).cuda()

    # Parameters
    batch_size = n.size(0)
    max_digit = 2 + 1
    sum_terms = sum_terms + 1

    # Prepare indices in a vectorized way
    k_values = torch.arange(1, sum_terms, dtype=torch.float32).view(1, 1, -1).cuda()  # Shape (1, 1, sum_terms - 1)
    d_values = torch.arange(max_digit, dtype=torch.float32).view(1, -1, 1).cuda()        # Shape (1, max_digit, 1)

    # Reshape n for broadcasting across digits and sum terms
    n = n.view(batch_size, 1, 1).cuda()  # Shape (batch, 1, 1)

    # Calculate A in a fully vectorized way
    A = torch.sin(2 * torch.pi * k_values * n / 10**d_values) / k_values  # Shape (batch, max_digit, sum_terms - 1)

    # Summations with the computed A, applied across the batch
    n_ones = torch.relu(4.5 + (1 / torch.pi) * (A[:, 0, :] - 10 * A[:, 1, :]).sum(dim=1))
    n_tens = torch.relu(4.5 + (1 / torch.pi) * (A[:, 1, :] - 10 * A[:, 2, :]).sum(dim=1))

    return n_ones, n_tens

# Make sure similarity_threshold is picked such that the correct number has a score greater than this threshold, and everything 
#  else is smaller than the threshold
def query_digit(SP, d, similarity_threshold=0.5, T=0.1, exp_scalar=100, k=100):
    query = bind(SP, inverse_vectors[digits[d]].cuda())

    vs = (torch.stack(list(number_SPs.values())).reshape(-1, len(number_SPs), SP_dim).cuda() @ query.T).T.squeeze(-1)

    digit_values = torch.softmax(vs/T, dim=1)
    digit_scores = 1/exp_scalar*torch.log(torch.exp(vs*exp_scalar).sum(dim=1)) # LogSumExp
    digit_scores = torch.sigmoid(k * (digit_scores - similarity_threshold))
    modified_digit_values = digit_values * digit_scores.unsqueeze(1)

    nums = torch.arange(0, 10, dtype=torch.float32).cuda()

    return (nums * modified_digit_values).sum(axis=-1)

def fractional_encode(x):
    return torch.fft.ifft((torch.fft.fft(vocabulary["SP_x"])**x.view(-1, 1))).real


def add_SP(SP, similarity_threshold=0.5):
    #SP = make_tensor_unitary(SP)

    n1 = bind(SP, vocabulary_inverse["SP_n1"].cuda())
    n2 = bind(SP, vocabulary_inverse["SP_n2"].cuda())

    n3 = null(SP.shape).cuda()
    r  = null((SP.shape[0])).cuda()
    for d in digits.keys():
        digit_n1 = query_digit(n1, d)
        digit_n2 = query_digit(n2, d)
        #print(d, digit_n1.item(), digit_n2.item(), r.item())

        digit_n3, r = single_digit_addition_fourier(digit_n1, digit_n2, r)
        #print(digit_n3.item(), r.item(), "\n")

        n3 += bind(vocabulary[d], fractional_encode(digit_n3))

    return n3

num1 + num2 - decode_SP(add_SP(final_SP.cuda())).item(), decode_SP(add_SP(final_SP.cuda())).item()


def generate_dialog(complexity=8, samples=1, problem_type="addition"):
    #x = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
    x = np.random.randint(low=1, high=10**(complexity+1), size=samples)
    #y = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
    y = np.random.randint(low=1, high=10**(complexity+1), size=samples)

    #example_x1, example_y1 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
    example_x1, example_y1 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
    #example_x2, example_y2 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
    example_x2, example_y2 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
    example_x1, example_y1 = max(example_x1, example_y1), min(example_x1, example_y1)
    example_x2, example_y2 = max(example_x2, example_y2), min(example_x2, example_y2)

    dialog: List[Dialog] = []

    if type(problem_type) == type([]):
        problem_type = random.choice(problem_type)
    
    if problem_type == "random":
        problem_type = random.choice(["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"])

    for n in range(samples):
        dialog += [
            [
                {"role": "system", "content": "You are a math solving helper. Don't use any commas in your output, and always answer problems according to the format of previous answers."},
            ]
        ]

        x[n], y[n] = max(x[n], y[n]), min(x[n], y[n])
        if problem_type == "addition":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} plus {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1 + example_y1}"},
                {"role": "user", "content": f"What is {example_x2} plus {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2 + example_y2}"},
                {"role": "user", "content": f"What is {x[n]} plus {y[n]}?"},
            ]

        elif problem_type == "multiplication":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} times {example_y1} mod {10**(complexity+1)}?"},
                {"role": "assistant", "content": f"{(example_x1 * example_y1) % 10**(complexity+1)}"},
                {"role": "user", "content": f"What is {example_x2} times {example_y2} mod {10**(complexity+1)}?"},
                {"role": "assistant", "content": f"{(example_x2 * example_y2) % 10**(complexity+1)}"},
                {"role": "user", "content": f"What is {x[n]} times {y[n]} mod {10**(complexity+1)}?"},
            ]

        elif problem_type == "division":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} // {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1//example_y1}"},
                {"role": "user", "content": f"What is {example_x2} // {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2//example_y2}"},
                {"role": "user", "content": f"What is {x[n]} // {y[n]}?"},
            ]

        elif problem_type == "modulo":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} mod {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1 % example_y1}"},
                {"role": "user", "content": f"What is {example_x2} mod {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2 % example_y2}"},
                {"role": "user", "content": f"What is {x[n]} mod {y[n]}?"},
            ]

        elif problem_type == "gcd":
            dialog[n] += [
                {"role": "user", "content": f"What is the GCD of {example_x1} and {example_y1}?"},
                {"role": "assistant", "content": f"{np.gcd(example_x1, example_y1)}"},
                {"role": "user", "content": f"What is the GCD of {example_x2} and {example_y2}?"},
                {"role": "assistant", "content": f"{np.gcd(example_x2, example_y2)}"},
                {"role": "user", "content": f"What is the GCD of {x[n]} and {y[n]}?"},
            ]

        elif problem_type == "lcm":
            dialog[n] += [
                {"role": "user", "content": f"What is the LCM of {example_x1} and {example_y1} mod {10**(complexity+1)}?"},
                {"role": "assistant", "content": f"{np.lcm(example_x1, example_y1) % 10**(complexity+1)}"},
                {"role": "user", "content": f"What is the LCM of {example_x2} and {example_y2} mod {10**(complexity+1)}?"},
                {"role": "assistant", "content": f"{np.lcm(example_x2, example_y2) % 10**(complexity+1)}"},
                {"role": "user", "content": f"What is the LCM of {x[n]} and {y[n]} mod {10**(complexity+1)}?"},
            ]

        elif problem_type == "square_mod":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1}^2 mod {example_y1}?"},
                {"role": "assistant", "content": f"{(example_x1)**2 % example_y1}"},
                {"role": "user", "content": f"What is {example_x2}^2 mod {example_y2}?"},
                {"role": "assistant", "content": f"{(example_x2)**2 % example_y2}"},
                {"role": "user", "content": f"What is {x[n]}^2 mod {y[n]}?"},
            ]

        elif problem_type == "bitwise_and":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} AND {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1 & example_y1}"},
                {"role": "user", "content": f"What is {example_x2} AND {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2 & example_y2}"},
                {"role": "user", "content": f"What is {x[n]} AND {y[n]}?"},
            ]

        elif problem_type == "bitwise_xor":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} XOR {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1 ^ example_y1}"},
                {"role": "user", "content": f"What is {example_x2} XOR {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2 ^ example_y2}"},
                {"role": "user", "content": f"What is {x[n]} XOR {y[n]}?"},
            ]
        elif problem_type == "bitwise_or":
            dialog[n] += [
                {"role": "user", "content": f"What is {example_x1} OR {example_y1}?"},
                {"role": "assistant", "content": f"{example_x1 | example_y1}"},
                {"role": "user", "content": f"What is {example_x2} OR {example_y2}?"},
                {"role": "assistant", "content": f"{example_x2 | example_y2}"},
                {"role": "user", "content": f"What is {x[n]} OR {y[n]}?"},
            ]


    return dialog, x, y, problem_type

def decode_digits(SP, n=SP_n1, verbose=False):
    if n == 0: 
        n == SP_n1
    decoded_values = []
    for d in digits.keys():
        dv = (query_digit(bind(SP, inverse_vectors[n].cuda()), d).item())
        if verbose:
            print(d, '\t', round(dv, 3))
        decoded_values += [dv]
    return np.array(decoded_values)

def decode_digits_tensor(SP, SP_n=SP_n1):
    decoded_values = []
    for d in digits.keys():
        dv = query_digit(bind(SP, inverse_vectors[SP_n].cuda()), d)
        decoded_values += [dv]
    decoded_values = torch.stack(decoded_values).T    
    return decoded_values

def decode_problem_type(SP, verbose=False):
    problem_type_SPs = []
    for p in possible_problems:
        problem_type_SPs += [vocabulary[p].flatten()]
    problem_type_SPs = torch.stack(problem_type_SPs)
    problem_type_labels = [possible_problems[i] for i in (problem_type_SPs @ SP.T.float()).T.argmax(axis=1)]
    if verbose:
        print(problem_type_labels)
    return np.array(problem_type_labels)


def digit_error(predictions, labels, error_per_digit=False, verbose=0):
    errors = []
    if error_per_digit:
        # First index corresponds to highest digit, last index corresponds to lowest digit
        digit_errors = np.zeros(complexity+ 1)

    for prediction, label in zip(predictions, labels):
        # Convert prediction and label to strings for digit-wise comparison
        pred_str = str(prediction)
        label_str = str(label)

        # Align digits by value (right-align with padding on the left)
        if verbose == 2:
            print("Predicted and actual numbers:", pred_str, label_str)
        max_length = max(len(pred_str), len(label_str))
        pred_str = pred_str.zfill(max_length)
        label_str = label_str.zfill(max_length)

        # Count incorrect digits
        
        
        incorrect_count = 0
        n = 0
        for pred_digit, label_digit in zip(pred_str, label_str):
            if verbose == 2:
                print(f"Digit Number {n}:", pred_digit, label_digit)
            if pred_digit != label_digit:
                incorrect_count += 1
                if error_per_digit:
                    digit_errors[n] += 1
            n += 1
            if n == (complexity + 1):
                break

        # Append the error for this pair
        errors.append(incorrect_count)

    if error_per_digit:
        return np.array(errors), digit_errors
    else:
        return np.array(errors)
    

class Encoder(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.encoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        self.layer_id = layer_id

    def forward(self, x):
        out = self.encoder_layer(x)
        return out

# class Decoder(nn.Module):
#     def __init__(self, layer_id, input_dim, output_dim, bias=False, dtype=torch.bfloat16):
#         super().__init__()
#         self.decoder_layer = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
#         self.layer_id = layer_id

#     def forward(self, x):
#         out = self.decoder_layer(x)
#         return out

# class Encoder_Deep(nn.Module):
#     def __init__(self, layer_id, input_dim, output_dim, hidden_dim, bias=False, dtype=torch.bfloat16):
#         super().__init__()
#         self.encoder_layer_1 = nn.Linear(input_dim,  hidden_dim, bias=bias, dtype=dtype)
#         self.encoder_layer_2 = nn.Linear(hidden_dim, output_dim, bias=bias, dtype=dtype)
#         self.activation = nn.ReLU()
#         self.layer_id = layer_id

#     def forward(self, x):
#         x   = self.activation(self.encoder_layer_1(x))
#         out = self.encoder_layer_2(x)
#         return out
# class Decoder_Deep(nn.Module):
#     def __init__(self, layer_id, input_dim, output_dim, hidden_dim, bias=False, dtype=torch.bfloat16):
#         super().__init__()
#         self.decoder_layer_1 = nn.Linear(input_dim,  hidden_dim, bias=bias, dtype=dtype)
#         self.decoder_layer_2 = nn.Linear(hidden_dim, output_dim, bias=bias, dtype=dtype)
#         self.activation = nn.ReLU()
#         self.layer_id = layer_id

#     def forward(self, x):
#         x   = self.activation(self.decoder_layer_1(x))
#         out = self.decoder_layer_2(x)
#         return out

# class EncoderDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, label

def gather_h_stacks(dialog_data):
    #response_data = []
    dialogs = dialog_data[0]
    x       = dialog_data[1]
    y       = dialog_data[2]
    problem_type = dialog_data[3]

    h_stack, list_of_probs, out_tokens = episode(dialogs, temperature=temperature,
                                                 inference_mode=self.model.forward, 
                                                 max_decoding_length=1,
                                                 )
    
    # shape of h_stack is [num_layers, batch_size, num_tokens, hidden_dm], per output token
    
    correct_sps = []
    for n in range(len(x)):
        correct_sp   = generate_SP(x[n], y[n], problem_type).to(torch.bfloat16)
        correct_sps += [correct_sp.flatten()]
    correct_sps = torch.stack(correct_sps)
    
    # return h_stack[0], since we are not concerned with the LLM output in this experiment
    return h_stack[0], correct_sps

def gather_sps(dialog_data):
    x       = dialog_data[1]
    y       = dialog_data[2]

    correct_sps = []
    for n in range(len(x)):
        correct_sp   = generate_SP(x[n], y[n]).to(torch.bfloat16)
        correct_sps += [correct_sp.flatten()]
    correct_sps = torch.stack(correct_sps)
    
    return correct_sps


train_data_rounds = 10000
test_data_rounds  = 1000

# Final number of items will be LLM batch size times train/test_data_rounds

save_frequency = 20

layer_numbers = torch.arange(0, 33)
#layer_numbers = torch.arange(12, 24)
complexity  = 3 #10 # Complexity of problems to ask, represented by number of digits + 1 (of x and y)
n_samples   = max_batch_size # should be less or equal to  than params.max_batch_size

problem_type = ["multiplication",  "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"]

if type(problem_type) == type([]):
    problem_str = "_".join(problem_type)
else:
    problem_str = problem_type

tokens_to_keep = 'all'
calculate_end_index = False

save_dir = f"gathered_data_{complexity}_complexity_{tokens_to_keep}_tokens_kept_{calculate_end_index}_cei_{train_data_rounds}_train_rounds_{test_data_rounds}_test_rounds_{problem_str}"
os.makedirs(save_dir, exist_ok=True)

model_dim = 4096

encoder_training_batch_size = 512
training_epochs = 10000
learning_rate = 1e-3 # Base learning rate, modified by learning_rate_reduction_factors
#learning_rate_reduction_factors = {25: 0.5, 50:  0.5, 100: 0.1, 250: .4}
learning_rate_reduction_factors = {50: 0.5, 100:  0.5, 250: 0.1, 500: .4}


def get_dialog_indices(dialog, calculate_end_index=False):
    start_indices = []
    end_indices   = []
    for i in range(len(dialog)):
        # Find the final occurance of user chat (which is the question being asked to the LLM)
        start_index = len(generator.parse_chat(dialog)[i]) - generator.parse_chat(dialog)[i][::-1].index(882) + 2
        # The final token position to save
        if not calculate_end_index:
            end_index   = -1 # If end_index is -1, use all tokens up till the end, otherwise calculate based on eot token
        else:
            end_index   = len(generator.parse_chat(dialog)[i]) - generator.parse_chat(dialog)[i][::-1].index(128009) - 1
        start_indices += [start_index]
        end_indices   += [end_index]

    return start_indices, end_indices


def generate_and_save_data(rounds, train, save_frequency, self, complexity, n_samples, problem_type,
                           tokens_to_keep=1, calculate_end_index=False, verbose=True):

    if train:
        h_path  = 'h_stack_round_'
        sp_path = 'correct_sps_round_'
    else:
        h_path  = 'testing_h_stack_round_'
        sp_path = 'testing_correct_sps_round_'

    h_stacks = []
    numbers  = []

    max_h_stack_tokens = 0

    # Save data per round to avoid keeping it in memory
    for r in range(rounds+1):
        if not r % save_frequency and r:
            if verbose:
                print("On Round Number:", r)
            numbers_stacked = torch.stack(numbers)
            max_tok_length = max([h.shape[1] for h in h_stacks])
            if tokens_to_keep == "all":
                h_stacked = torch.stack([F.pad(h, (0, 0, 0, 0, max_tok_length - h.shape[1], 0, 0, 0))[0,:,:,:,] if max_tok_length != h.shape[1] 
                                         else h[0,:,:,:,] for h in h_stacks])
            else:
                h_stacked = torch.stack(h_stacks)
                h_stacked = h_stacked.view(-1, tokens_to_keep, self.model.params.dim, self.model.params.n_layers+1)
            # When saved, the shape of h_stacked is (batch, num_tokens, hidden_dim, n_layers)
            torch.save(h_stacked, os.path.join(save_dir, f"{h_path}{r}.pt"))
            torch.save(numbers_stacked, os.path.join(save_dir, f"{sp_path}{r}.pt"))
            h_stacks = []
            numbers  = []
            if r == train_data_rounds:
                break

        # Generate dialog data and gather 'h_stack' and 'correct_sps'
        dialog_data = generate_dialog(complexity=complexity, samples=n_samples, problem_type=problem_type)
        h_stack, correct_sps = gather_h_stacks(dialog_data)
        # shape of h_stack is n_layers, batch, num_tokens, hiddem_dim.
        
        if tokens_to_keep == "all":
            # Dialog_data[0] is the dialogs 
            start_indices, end_indices = get_dialog_indices(dialog_data[0], calculate_end_index=calculate_end_index)
            max_h_stack_tokens = max(max_h_stack_tokens, max(start_indices)) # increase max_h_stack
            if calculate_end_index:
                h_stacks += [h_stack[:,b:b+1,start_indices[b]:end_indices[b],:,].permute((1, 2, 3, 0)) 
                             for b in range(h_stack.shape[1])]
            else:
                h_stacks += [h_stack[:,b:b+1,start_indices[b]:,:,].permute((1, 2, 3, 0)) 
                             for b in range(h_stack.shape[1])]
        else:
            h_stacks += [h_stack[:,:,-tokens_to_keep:,:,].permute((1, 2, 3, 0))] 
        # shape of h_stacks[-1] is batch, num_tokens, hiddem_dim, n_layers. len of it is number of runs
        numbers += correct_sps

def generate_data_loaders(train, save_dir, data_rounds, save_frequency, layer_numbers, restrict_dataset=None, 
                          tokens_to_keep=1, batch_size=512, verbose=False):
    if train:
        h_path  = 'h_stack_round_'
        sp_path = 'correct_sps_round_'
        shuffle = True
    else:
        h_path  = 'testing_h_stack_round_'
        sp_path = 'testing_correct_sps_round_'
        shuffle = False

    # Load data for each layer to create data loaders
    encoder_data_loaders = []

    for n_layer in layer_numbers:
        if verbose:
            print("On Layer Number:", n_layer.item())
        h_layer_data = []
        correct_sps_data = []

        # Load each round's data from disk
        if restrict_dataset:
            runs = restrict_dataset
        else:
            runs = data_rounds
        for r in range(runs+1):
            if not r % save_frequency and r:
                if verbose:
                    print("On Round Number:", r)
                h_stack = torch.load(os.path.join(save_dir, f"{h_path}{r}.pt"))
                correct_sps = torch.load(os.path.join(save_dir, f"{sp_path}{r}.pt"))

                # Collect data for the specific layer
                if   tokens_to_keep == "all":
                    h_layer_data.append(h_stack[:,                :, :, n_layer])
                elif tokens_to_keep == 1:
                    h_layer_data.append(h_stack[:,               -1, :, n_layer])
                else:
                    h_layer_data.append(h_stack[:, -tokens_to_keep:, :, n_layer])

                correct_sps_data.append(correct_sps)

        # Stack data for the current layer
        if tokens_to_keep == "all":
            max_tok_length = max([h.shape[1] for h in h_layer_data])
            h_layer_stacked = torch.cat([F.pad(h, (0, 0, max_tok_length - h.shape[1], 0, 0, 0)) for h in h_layer_data], dim=0)
        else:
            h_layer_stacked = torch.cat(h_layer_data, dim=0)
        numbers_stacked = torch.cat(correct_sps_data, dim=0)

        # Create `EncoderDataset` and `DataLoader` for the current layer
        encoder_training_data = EncoderDataset(h_layer_stacked.cuda(), numbers_stacked.cuda())
        gpu_generator = torch.Generator(device='cuda')
        gpu_generator.manual_seed(42)

        encoder_data_loader = DataLoader(
            encoder_training_data,
            batch_size=batch_size,
            shuffle=shuffle,
            #generator=gpu_generator,
        )
        encoder_data_loaders.append(encoder_data_loader)
        
    return encoder_data_loaders


if generate_data:
    generate_and_save_data(rounds=train_data_rounds, train=True,  save_frequency=save_frequency, self=self, 
                           complexity=complexity, n_samples=n_samples, problem_type=problem_type,
                           tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True)
    print("Training data gathering completed and saved to disk.")

    generate_and_save_data(rounds=test_data_rounds,  train=False, save_frequency=save_frequency, self=self, 
                           complexity=complexity, n_samples=n_samples, problem_type=problem_type,
                           tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True)
    print("Testing data gathering completed and saved to disk.")
    wandb.finish()

else:
    encoder_data_loaders = generate_data_loaders(train=True, save_dir=save_dir, data_rounds=train_data_rounds, 
                                                 save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=1000, 
                                                 tokens_to_keep=tokens_to_keep, batch_size=encoder_training_batch_size, verbose=True)
    print("Training data loaders for each layer have been created.")
    testing_encoder_data_loaders = generate_data_loaders(train=False, save_dir=save_dir, data_rounds=test_data_rounds, 
                                                         save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=1000, 
                                                         tokens_to_keep=tokens_to_keep, batch_size=encoder_training_batch_size, verbose=True)
    print("Testing data loaders for each layer have been created.")

    class LastTokenTransformer(nn.Module):
        def __init__(self, layer_id, data_dim, output_dim, num_layers=4,
                    num_heads=8, hidden_dim=512, dropout=0.1, dtype=torch.bfloat16):
            """
            Transformer model that processes LLM hidden states and outputs a semantic vector.

            Args:
                data_dim (int): Dimension of the input hidden states.
                output_dim (int): Dimension of the final semantic output.
                num_layers (int): Number of transformer encoder layers.
                num_heads (int): Number of attention heads.
                hidden_dim (int): Size of the feedforward hidden layer.
                dropout (float): Dropout rate.
            """
            super().__init__()

            self.layer_id = layer_id
            self.data_dim = data_dim
            self.output_dim = output_dim

            # Input projection layer (embedding input into model hidden size)
            self.input_proj = nn.Linear(data_dim, hidden_dim, dtype=dtype)

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                dtype=dtype
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection layer
            self.output_proj = nn.Linear(hidden_dim, output_dim, dtype=dtype)

        def forward(self, x):
            """
            Args:
                x (Tensor): Input tensor of shape (batch, sequence_number, data_dim)

            Returns:
                Tensor: Output tensor of shape (batch, output_dim)
            """
            batch_size, seq_len, _ = x.shape

            # Project input to hidden dimension
            x = self.input_proj(x)  # Shape: (batch, seq_len, hidden_dim)

            # Pass through transformer encoder
            x = self.transformer_encoder(x)  # Shape: (batch, seq_len, hidden_dim)

            # Select the last token's hidden state
            last_token = x[:, -1, :]  # Shape: (batch, hidden_dim)

            # Output projection
            out = self.output_proj(last_token)  # Shape: (batch, output_dim)

            return out

    encoders = torch.nn.ModuleList()
    for layer_id in layer_numbers:
        if tokens_to_keep == 1:
            #layer_encoder = Encoder_Deep(layer_id, model_dim, SP_dim, model_dim*4).to(device)
            layer_encoder = Encoder(layer_id, model_dim, SP_dim, ).to(device)
        else:
            layer_encoder = LastTokenTransformer(layer_id, model_dim, SP_dim, num_layers=2, hidden_dim=512).to(device)
        encoders.append(layer_encoder)#, dtype=torch.float32))

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_trainable_parameters(layer_encoder) # Per Layer

    optimizers   = [optim.Adam(encoders[n].parameters(), lr=learning_rate) for n in range(len(layer_numbers))]
    criterion = nn.MSELoss()
    losses = np.zeros((len(layer_numbers), training_epochs))
    running_losses = np.zeros((len(layer_numbers)))
    for i in range(training_epochs):
        if i in learning_rate_reduction_factors.keys():
            for param_group in optimizers[n].param_groups:
                param_group['lr'] = param_group['lr'] * learning_rate_reduction_factors[i]  # Set new learning rate
                print("Learning Rate changed to:", param_group['lr'])

        for n, n_layer in enumerate(layer_numbers):
            encoders[n].train()
            running_loss = 0
            total_norm = 0.0
            for batch_idx, (data, labels) in enumerate(encoder_data_loaders[n]):
                model_pred = encoders[n](data)
                loss = torch.sqrt(criterion(model_pred, labels))
                loss.backward()
                tn = 0
                for p in encoders[n].parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        tn += param_norm.item() ** 2
                total_norm += tn ** 0.5
                optimizers[n].step()
                optimizers[n].zero_grad()
                running_loss += loss.item()
            running_loss /= (batch_idx + 1)
            running_losses[n] = running_loss
            total_norm   /= (batch_idx + 1)
            losses[n][i] = running_loss
        if not i % 100:
            print("Epoch:", i, "Running Loss:", running_losses.mean(), f"\tTotal gradient norm: {total_norm}")


    #errors_per_pt = {layer_num.item() : {pt: [] for pt in possible_problems} for n, layer_num in enumerate(layer_numbers)}
    errors_per_pt = {pt: {layer_num.item(): [] for n, layer_num in enumerate(layer_numbers)} for pt in problem_type}

    SP_predictions = []
    label_SPs      = []
    rows_to_print  = 0
    verbose = 1 # 0, 1, 2
    errors = np.zeros(len(layer_numbers))
    lowest_error_layer = 0
    lowest_error       = complexity + 1
    lowest_pt_error    = np.inf
    calculate_digit_error = True # Measure total number of errors per digit
    per_digit_errors = np.zeros((len(layer_numbers), complexity+1))
    for n, layer in enumerate(layer_numbers):
        row_count = 0
        if verbose:
            print("--------- Layer", layer.item(), "---------")
        e = 0
        for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
            row_count += len(data)
            pred = encoders[n](data)
            decoded_n1 = [sum([round(i, 0) * 10**n for n, i in enumerate(decode_digits(pred[i].  type_as(vectors[SP_n1]), SP_n1))]) for i in range(pred.  shape[0])]
            decoded_n2 = [sum([round(i, 0) * 10**n for n, i in enumerate(decode_digits(pred[i].  type_as(vectors[SP_n2]), SP_n2))]) for i in range(pred.  shape[0])]
            actual_n1  = [sum([round(i, 0) * 10**n for n, i in enumerate(decode_digits(labels[i].type_as(vectors[SP_n1]), SP_n1))]) for i in range(labels.shape[0])]
            actual_n2  = [sum([round(i, 0) * 10**n for n, i in enumerate(decode_digits(labels[i].type_as(vectors[SP_n2]), SP_n2))]) for i in range(labels.shape[0])]
            decoded_problem_types = decode_problem_type(pred)
            actual_problem_types  = decode_problem_type(labels)
            if calculate_digit_error:
                n1_batch_error, n1_per_digit_error = digit_error(decoded_n1, actual_n1, error_per_digit=calculate_digit_error, verbose=verbose)
                n2_batch_error, n2_per_digit_error = digit_error(decoded_n2, actual_n2, error_per_digit=calculate_digit_error, verbose=verbose)
                batch_error = (n1_batch_error + n2_batch_error) / 2
                per_digit_errors[layer.item()] += (n1_per_digit_error + n2_per_digit_error) / 2 * len(data) / (test_data_rounds * max_batch_size)
            else:
                batch_error = digit_error(decoded_n1, actual_n1, verbose=verbose) + digit_error(decoded_n2, actual_n2, verbose=verbose)
            for k, curr_pt in enumerate(actual_problem_types):
                errors_per_pt[curr_pt][layer.item()] += [batch_error[k]]
            for r in range(rows_to_print):
                print("Decoded symbolic encodings: first number:",  decoded_n1[r], "second number:", decoded_n2[r])
                print("Actual           encodings: first number:",  actual_n1[r],  "second number:", actual_n2[r])
                print("Decoded problem type:", decoded_problem_types[r])
                print("Actual  problem type:", actual_problem_types[r])
                #print(" --------- Error:", decoded_n1[r]-actual_n1[r], decoded_n2[r]-actual_n2[r], )
            e += np.mean(digit_error(decoded_n1, actual_n1, verbose=verbose) + digit_error(decoded_n2, actual_n2, verbose=verbose)) / 2 * len(data) / (test_data_rounds * max_batch_size)
        errors[n] = e
        per_digit_errors[layer.item()] = per_digit_errors[layer.item()] / row_count
        problem_type_error = (decoded_problem_types != actual_problem_types).sum()
        if e < lowest_error:
            lowest_error_layer = layer
            lowest_error       = e
            lowest_pt_error    = problem_type_error

        print("Average Error:", np.mean(e), "digits out of", complexity+1)
        print("Average Problem Type Error:", problem_type_error, "out of", len(labels))
        # Divide per_digit_errors by row_count and by 2 in order to get per digit error
        print("Average Error Rate per Digit:", per_digit_errors[layer.item()])

    x_ticks = np.arange(layer_numbers[0].item(), layer_numbers[0].item() + len(layer_numbers), 1)  # Adjust the range as needed
    plt.plot(x_ticks, errors, marker=".")
    plt.xticks(x_ticks, rotation=75)
    plt.title("Error of Decoded Numbers (Testing Data)")
    plt.ylabel("Average Number of Incorrectly Decoded Digits")
    plt.xlabel("Layer Number")
    plt.grid(False)
    #plt.savefig("error_per_layer.png")
    wandb.log({f"Error of Decoded Numbers (Testing Data)": wandb.Image(plt)})  # Log to wandb
    plt.close()

    for pt in problem_type:
        x_ticks = np.arange(layer_numbers[0].item(), layer_numbers[0].item() + len(layer_numbers), 1)  # Adjust the range as needed
        pt_error = [np.mean(errors_per_pt[pt][ln]) for ln in errors_per_pt[pt]]
        plt.plot(x_ticks, pt_error, marker=".")
        plt.xticks(x_ticks, rotation=75)
        #plt.close()
    #plt.title(f"Error of Decoded Numbers per Problem Type (Testing Data)")
    plt.ylabel("Mean Absolute Decoding Error")
    plt.xlabel("Layer Number")
    plt.legend(problem_type)
    plt.grid(False)
    #plt.savefig("per_pt_error_per_layer.png")
    wandb.log({f"Mean Absolute Decoding Error": wandb.Image(plt)})  # Log to wandb
    plt.close()

    labels = ['Ones Digit Error Rate',
            'Tens Digit Error Rate',
            'Hundreds Digit Error Rate',
            'Thousands Digit Error Rate',
            'Ten Thousands Digit Error Rate',
            'Hundred Thousands Digit Error Rate',
            'Millions Digit Error Rate',
            'Ten Millions Digit Error Rate',
            'Hundred Millions Digit Error Rate',
            ]
    markers = ["o", "s", "^", 
            ".", "v", "*", 
            "<", ">", "1"]

    for n, digit in enumerate(np.array(per_digit_errors).T):
        plt.plot(layer_numbers, digit, label=labels[n], marker=markers[n])

    # # Plotting
    # #plt.figure(figsize=(12, 6))
    # plt.plot(layers, digit1, label='Hundreds Digit Error Rate', marker='o')
    # plt.plot(layers, digit2, label='Tens Digit Error Rate', marker='s')
    # plt.plot(layers, digit3, label='Ones Digit Error Rate', marker='^')

    # Adding labels, title, and legend
    plt.xlabel('Layer Number')
    plt.ylabel('Classification Error Rate')
    plt.title('Per Digit Loss Per Layers')
    plt.legend()
    plt.grid(True)
    #plt.savefig("per_digit_error_per_layer.png")
    wandb.log({f"Per Digit Loss Per Layers": wandb.Image(plt)})  # Log to wandb
    plt.close()

    print("Minimum Error:", lowest_error, "and problem type error:", problem_type_error, "at layer", lowest_error_layer.item(), ", Current running loss:", running_losses[lowest_error_layer.item()])

    SP_predictions = []
    label_SPs      = []
    max_rows = 100
    verbose = False
    train_errors = torch.zeros((len(layer_numbers), len(digits)))
    for n, layer in enumerate(layer_numbers):
        encoders[n] = encoders[n].to(device).eval()

    with torch.no_grad():
        for n, layer in enumerate(layer_numbers):
            row_count = 0
            for batch_idx, (data, labels) in enumerate(encoder_data_loaders[n]):
                pred = encoders[n](data)
                SP_predictions += [pred]
                label_SPs += [labels]
                actual_digits_n1    = decode_digits_tensor(labels.to(torch.float32), SP_n1)
                actual_digits_n2    = decode_digits_tensor(labels.to(torch.float32), SP_n2)
                predicted_digits_n1 = decode_digits_tensor(pred.to(torch.float32), SP_n1)
                predicted_digits_n2 = decode_digits_tensor(pred.to(torch.float32), SP_n2)
                digit_errors = (actual_digits_n1-predicted_digits_n1).mean(axis=0)
                #print(digit_errors.mean().cpu().item())
                train_errors[n] += digit_errors.cpu() * len(data)
                row_count += len(data)
        train_errors[n] /= row_count

    plt.plot(train_errors.mean(axis=1), marker=".")
    x_ticks = np.arange(0, len(layer_numbers), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Error of Decoded Numbers (Training Data)")
    plt.ylabel("Mean Absolute Decoding Error")
    plt.xlabel("Layer Number")
    wandb.log({f"Error of Decoded Numbers (Training Data)": wandb.Image(plt)})  # Log to wandb
    plt.close()

    for n, d in enumerate(digits):
        plt.plot(train_errors[:,n], marker=".")
        x_ticks = np.arange(0, len(layer_numbers), 1)  # Adjust the range as needed
        plt.xticks(x_ticks, rotation=75)
        plt.title(f"Digit {n} Error of Decoded Numbers (Training Data)")
        plt.ylabel("Mean Absolute Decoding Error")
        plt.xlabel("Layer Number")
        wandb.log({f"Digit {n} Error of Decoded Numbers (Training Data)": wandb.Image(plt)})  # Log to wandb
        plt.close()

    SP_predictions = []
    label_SPs      = []
    max_rows = 100
    verbose = False
    test_errors = torch.zeros((len(layer_numbers), len(digits)))
    for n, layer in enumerate(layer_numbers):
        encoders[n] = encoders[n].to(device).eval()

    with torch.no_grad():
        for n, layer in enumerate(layer_numbers):
            row_count = 0
            for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
                pred = encoders[n](data)
                SP_predictions += [pred]
                label_SPs += [labels]
                actual_digits_n1    = decode_digits_tensor(labels.to(torch.float32), SP_n1)
                actual_digits_n2    = decode_digits_tensor(labels.to(torch.float32), SP_n2)
                predicted_digits_n1 = decode_digits_tensor(pred.to(torch.float32), SP_n1)
                predicted_digits_n2 = decode_digits_tensor(pred.to(torch.float32), SP_n2)
                digit_errors = (actual_digits_n1-predicted_digits_n1).mean(axis=0)
                #print(digit_errors.mean().cpu().item())
                test_errors[n] += digit_errors.cpu() * len(data)
                row_count += len(data)
        test_errors[n] /= row_count

    plt.plot(test_errors.mean(axis=1), marker=".")
    x_ticks = np.arange(0, len(layer_numbers), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Error of Decoded Numbers (Testing Data)")
    plt.ylabel("Mean Absolute Decoding Error")
    plt.xlabel("Layer Number")
    wandb.log({f"Error of Decoded Numbers (Testing Data)": wandb.Image(plt)})  # Log to wandb
    plt.close()

    plt.plot(losses.T[:i].mean(axis=0)[:-1], marker=".")
    x_ticks = np.arange(0, len(losses.T.mean(axis=0)[:-1]), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Average RMSE Loss vs Layer Number")
    plt.ylabel("Average RMSE Loss")
    plt.xlabel("Layer Number")
    wandb.log({f"Average RMSE Loss vs Layer Number": wandb.Image(plt)})  # Log to wandb
    plt.close()

    # training_start = 100

    # tensor = losses[:,training_start:i]
    # x = np.arange(tensor.shape[0])
    # y = np.arange(tensor.shape[1])
    # X, Y = np.meshgrid(x, y)

    # # Transpose the tensor to match the meshgrid dimensions
    # Z = tensor.T  # Shape becomes (2000, 32)

    # # Create an interactive 3D surface plot
    # fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='inferno')])

    # # Customize layout
    # fig.update_layout(
    #     title='Loss vs Layer Number and Training Epoch',
    #     scene=dict(
    #         xaxis_title='Layer Number',
    #         yaxis_title='Training Epoch',
    #         zaxis_title='Loss',
    #     ),
    # )

    # # Show the plot
    # fig.show()

    plt.plot(losses.T[:i+1][-1,:], marker=".")
    x_ticks = np.arange(0, len(losses.T.mean(axis=0)[:-1]), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Final RMSE Loss vs Layer Number")
    plt.ylabel("Final RMSE Loss")
    plt.xlabel("Layer Number")
    wandb.log({f"Final RMSE Loss vs Layer Number": wandb.Image(plt)})  # Log to wandb
    plt.close()


    plt.plot(losses.mean(axis=0)[:i], marker=".")
    plt.title("Average RMSE Loss Per Epoch")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    wandb.log({f"Average RMSE Loss Per Epoch": wandb.Image(plt)})  # Log to wandb
    plt.close()

    ####################################################################################################

    for n, n_layer in enumerate(layer_numbers):
        for param in encoders[n].parameters():
            param.requires_grad = False



    decoders = torch.nn.ModuleList()
    for layer_id in layer_numbers:
        #layer_decoder = Decoder_Deep(layer_id, SP_dim, model_dim, model_dim*2).to(device)
        layer_decoder = Decoder(layer_id, SP_dim, model_dim).to(device)
        decoders.append(layer_decoder)#, dtype=torch.float32))

    decoding_training_epochs = 10000
    decoding_learning_rate = 1e-3 # Base learning rate, modified by learning_rate_reduction_factors
    decoding_learning_rate_reduction_factors = {10: 0.1, 25:  0.5, 100: 0.5, 250: .4}

    decoding_optimizers   = [optim.Adam(decoders[n].parameters(), lr=decoding_learning_rate) for n in range(len(layer_numbers))]
    decoding_criterion = nn.MSELoss()
    decoding_losses = np.zeros((len(layer_numbers), decoding_training_epochs))
    decoding_running_losses = np.zeros((len(layer_numbers)))
    for j in range(decoding_training_epochs):
        if j in decoding_learning_rate_reduction_factors.keys():
            for param_group in decoding_optimizers[n].param_groups:
                param_group['lr'] = param_group['lr'] * decoding_learning_rate_reduction_factors[j]  # Set new learning rate
                print("Learning Rate changed to:", param_group['lr'])
        for n, n_layer in enumerate(layer_numbers):
            decoding_running_loss = 0
            total_norm = 0.0
            for batch_idx, (data, labels) in enumerate(encoder_data_loaders[n]):
                latent_representation = encoders[n](data)
                
                #std = torch.exp(0.5 * latent_logvar)  # Compute standard deviation
                #epsilon = torch.randn_like(std)      # Sample noise
                #latent_representation = latent_representation + epsilon

                predicted_hidden_state = decoders[n](latent_representation)
                
                if tokens_to_keep != 1:
                    target = data[:,-1,:]
                else:
                    target = data

                loss = torch.sqrt(criterion(predicted_hidden_state, target))
                loss.backward()
                tn = 0
                for p in decoders[n].parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        tn += param_norm.item() ** 2
                total_norm += tn ** 0.5
                decoding_optimizers[n].step()
                decoding_optimizers[n].zero_grad()
                decoding_running_loss += loss.item()
            decoding_running_loss /= (batch_idx + 1)
            decoding_running_losses[n] = decoding_running_loss
            total_norm   /= (batch_idx + 1)
            decoding_losses[n][j] = decoding_running_loss
        if not j % 5 and j:
            print("Epoch:", j, "Running Loss:", decoding_running_losses.mean(), f"\tTotal gradient norm: {total_norm}")


    plt.plot(decoding_losses.mean(axis=0)[:j], marker=".")
    plt.title("Average Decoder RMSE Loss Per Epoch")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    wandb.log({f"Average Decoder RMSE Loss Per Epoch": wandb.Image(plt)})  # Log to wandb
    plt.close()

    plt.plot(decoding_losses.T[:i].mean(axis=0)[:-1], marker=".")
    x_ticks = np.arange(0, len(decoding_losses.T.mean(axis=0)[:-1]), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Average Decoder RMSE Loss vs Layer Number")
    plt.ylabel("Average RMSE Loss")
    plt.xlabel("Layer Number")
    wandb.log({f"Average Decoder RMSE Loss vs Layer Number": wandb.Image(plt)})  # Log to wandb
    plt.close()

    plt.plot(decoding_losses.T[:j+1][-1,:], marker=".")
    x_ticks = np.arange(0, len(decoding_losses.T.mean(axis=0)[:-1]), 1)  # Adjust the range as needed
    plt.xticks(x_ticks, rotation=75)
    plt.title("Final Decoder RMSE Loss vs Layer Number")
    plt.ylabel("Final RMSE Loss")
    plt.xlabel("Layer Number")
    wandb.log({f"Final Decoder RMSE Loss vs Layer Number": wandb.Image(plt)})  # Log to wandb
    plt.close()


    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(encoders.state_dict(), f"models/encoders_state_dict_{curr_date}.pth")
    torch.save(encoders,              f"models/encoders_{curr_date}.pth")
    print("Saved:", f"models/encoders_state_dict_{curr_date}.pth", "and", f"models/encoders_{curr_date}.pth")
    torch.save(decoders.state_dict(), f"models/decoders_state_dict_{curr_date}.pth")
    torch.save(decoders,              f"models/decoders_{curr_date}.pth")
    print("Saved:", f"models/decoders_state_dict_{curr_date}.pth", "and", f"models/decoders_{curr_date}.pth")


    wandb.finish()

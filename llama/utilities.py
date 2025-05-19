from __future__ import annotations

import json
import torch
import numpy as np
import random
import os
import pandas as pd
import sys
import math
import argparse
from pathlib import Path
import datetime
import wandb

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import num2words as n2w
from word2number import w2n

import plotly.graph_objects as go

from typing import List, Optional
import fire

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from typing import List

from llama import Dialog
from llama.generation import sample_top_p


def append_dialog_batch_to_file(dialog_batch, filepath):
    """Append each dialog in the batch (a list of list of dicts) to a JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for dialog in dialog_batch:
            json.dump(dialog, f)
            f.write('\n')


class EncoderDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


# def generate_dialog(complexity=8, # complexity + 1 is the maximum number of digits the generated numbers (input and output) can be
#                     samples=1, # Number of samples to generate. Note all generated samples will be of the same problem type
#                     problem_type="addition", # Problem type to generate. If set to "random" or a list of problem types, randomly select a problem type from either the entire set of possible problems or the specified subset, repsectively
#                     cot=False,  # If true, use CoT prompting
#                     string_nums=False, # If true, represent numbers as words (e.g., two hundred and one)
#                     limit_solution_digits=True, # If True, certain problem types whose solutions have more digits than their inputs will have their solutions truncated (via solution mod 10^(complexity + 1))
#                     modify_question_format=False, # If True, modify the manner in which questions are asked (e.g., ask "What is 12 * 32" or "Multiply 12 and 32" instead of "What is 12 times 32")
#                     ):
#     #x = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
#     x = np.random.randint(low=1, high=10**(complexity+1), size=samples)
#     #y = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
#     y = np.random.randint(low=1, high=10**(complexity+1), size=samples)
    
#     temp_x = []
#     temp_y = []
#     for n in range(samples):
#         x[n], y[n] = max(x[n], y[n]), min(x[n], y[n])
#         if string_nums:
#             temp_x += [n2w.num2words(x[n])]
#             temp_y += [n2w.num2words(y[n])]
#     if string_nums:
#         x, y, temp_x, temp_y = np.array(temp_x), np.array(temp_y), x, y

#     #example_x1, example_y1 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
#     example_x1, example_y1 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
#     #example_x2, example_y2 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
#     example_x2, example_y2 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
#     example_x1, example_y1 = max(example_x1, example_y1), min(example_x1, example_y1)
#     example_x2, example_y2 = max(example_x2, example_y2), min(example_x2, example_y2)
    
#     if string_nums:
#         example_x1, example_y1 = n2w.num2words(example_x1), n2w.num2words(example_y1)
#         example_x2, example_y2 = n2w.num2words(example_x2), n2w.num2words(example_y2)

#     if string_nums:
#         conv = lambda x: w2n.word_to_num(str(x))
#         conv_inv = lambda x: n2w.num2words(int(x))

#     else:
#         conv = lambda x: x
#         conv_inv = lambda x: x


#     dialog: List[Dialog] = []

#     if type(problem_type) == type([]):
#         problem_type = random.choice(problem_type)
    
#     if problem_type == "random":
#         problem_type = random.choice(["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"])

#     for n in range(samples):
#         if cot:
#             dialog += [
#                 [
#                     {"role": "system", "content": 
#                      "You are a math-solving assistant. Always explain your reasoning step by step. "
#                      "Regardless of the steps taken, ensure the final answer is clearly marked with 'Final Answer: x'."
#                     },
#                 ]
#             ]
#         else:
#             dialog += [
#                 [
#                     {"role": "system", "content": 
#                      "You are a math solving helper. Don't use any commas in your output, "
#                      "and always answer problems according to the format of previous answers."
#                     },
#                 ]
#             ]

#         if problem_type == "addition":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} plus {y[n]}?"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} plus {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1) + conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} plus {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2) + conv(example_y2))}"},
#                 ]

#                 if modify_question_format:
#                     # Define the list of different formats
#                     formats = [
#                         "{x} + {y}",
#                         "Work out {x} + {y}.",
#                         "Calculate {x} + {y}.",
#                         "What is {x} plus {y}?",
#                         "Add {x} and {y}.",
#                         "Sum of {x} and {y}.",
#                         "What is the sum of {x} and {y}?",
#                     ]
#                     # Randomly pick one format
#                     chosen_format = random.choice(formats)
#                     # Format the question
#                     question = chosen_format.format(x=x[n], y=y[n])
#                     dialog[n] += [{"role": "user", "content": question}]
#                 else:
#                     dialog[n] += [{"role": "user", "content": f"What is {x[n]} plus {y[n]}?"},]
#         elif problem_type == "multiplication":
#             if cot:
#                 if limit_solution_digits:
#                     dialog[n] += [
#                         {"role": "user", "content": f"Solve the following problem step by step: " 
#                         f"What is {x[n]} times {y[n]} mod {10**(complexity+1)}?"},
#                     ]
#                 else:
#                     dialog[n] += [
#                         {"role": "user", "content": f"Solve the following problem step by step: " 
#                         f"What is {x[n]} times {y[n]}?"},
#                     ]
#             else:
#                 if limit_solution_digits:
#                     dialog[n] += [
#                         {"role": "user", "content": f"What is {example_x1} times {example_y1} mod {10**(complexity+1)}?"},
#                         {"role": "assistant", "content": f"{conv_inv((conv(example_x1) * conv(example_y1)) % 10**(complexity+1))}"},
#                         {"role": "user", "content": f"What is {example_x2} times {example_y2} mod {10**(complexity+1)}?"},
#                         {"role": "assistant", "content": f"{conv_inv((conv(example_x2) * conv(example_y2)) % 10**(complexity+1))}"},
#                     ]

#                     if modify_question_format:
#                         # For modular multiplication, only slight variations make sense
#                         formats = [
#                             "What is {x} times {y} mod {mod}?",
#                             "Calculate {x} * {y} modulo {mod}.",
#                             "Work out {x} times {y} mod {mod}.",
#                             "Find the result of {x} multiplied by {y} modulo {mod}.",
#                         ]
#                         chosen_format = random.choice(formats)
#                         question = chosen_format.format(x=x[n], y=y[n], mod=10**(complexity+1))
#                         dialog[n] += [{"role": "user", "content": question}]
#                     else:
#                         dialog[n] += [{"role": "user", "content": f"What is {x[n]} times {y[n]} mod {10**(complexity+1)}?"}]

#                 else:
#                     dialog[n] += [
#                         {"role": "user", "content": f"What is {example_x1} times {example_y1}?"},
#                         {"role": "assistant", "content": f"{conv_inv((conv(example_x1) * conv(example_y1)))}"},
#                         {"role": "user", "content": f"What is {example_x2} times {example_y2}?"},
#                         {"role": "assistant", "content": f"{conv_inv((conv(example_x2) * conv(example_y2)))}"},
#                     ]

#                     if modify_question_format:
#                         # For normal multiplication
#                         formats = [
#                             "{x} * {y}",
#                             "Work out {x} * {y}.",
#                             "Calculate {x} * {y}.",
#                             "What is {x} times {y}?",
#                             "Multiply {x} and {y}.",
#                             "Product of {x} and {y}.",
#                             "What is the product of {x} and {y}?",
#                         ]
#                         chosen_format = random.choice(formats)
#                         question = chosen_format.format(x=x[n], y=y[n])
#                         dialog[n] += [{"role": "user", "content": question}]
#                     else:
#                         dialog[n] += [{"role": "user", "content": f"What is {x[n]} times {y[n]}?"}]


#         elif problem_type == "division":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} // {y[n]}?"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} // {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1)//conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} // {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2)//conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]} // {y[n]}?"},
#                 ]

#         elif problem_type == "modulo":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} mod {y[n]}?"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} mod {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1) % conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} mod {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2) % conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]} mod {y[n]}?"},
#                 ]

#         elif problem_type == "gcd":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is the GCD of {x[n]} and {y[n]}?"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is the GCD of {example_x1} and {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(example_x1), conv(example_y1)))}"},
#                     {"role": "user", "content": f"What is the GCD of {example_x2} and {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(example_x2), conv(example_y2)))}"},
#                     {"role": "user", "content": f"What is the GCD of {x[n]} and {y[n]}?"},
#                 ]

#         elif problem_type == "lcm":
#             if cot:
#                 if limit_solution_digits:
#                     dialog[n] += [
#                         {"role": "user", "content": f"Solve the following problem step by step: " 
#                         f"What is the LCM of {x[n]} and {y[n]} mod {10**(complexity+1)}?"},
#                     ]
#                 else:
#                     dialog[n] += [
#                         {"role": "user", "content": f"Solve the following problem step by step: " 
#                         f"What is the LCM of {x[n]} and {y[n]}?"},
#                     ]

#             else:
#                 if limit_solution_digits:
#                     dialog[n] += [
#                         {"role": "user", "content": f"What is the LCM of {example_x1} and {example_y1} mod {10**(complexity+1)}?"},
#                         {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x1), conv(example_y1)) % 10**(complexity+1))}"},
#                         {"role": "user", "content": f"What is the LCM of {example_x2} and {example_y2} mod {10**(complexity+1)}?"},
#                         {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x2), conv(example_y2)) % 10**(complexity+1))}"},
#                         {"role": "user", "content": f"What is the LCM of {x[n]} and {y[n]} mod {10**(complexity+1)}?"},
#                     ]
#                 else:
#                     dialog[n] += [
#                         {"role": "user", "content": f"What is the LCM of {example_x1} and {example_y1}?"},
#                         {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x1), conv(example_y1)))}"},
#                         {"role": "user", "content": f"What is the LCM of {example_x2} and {example_y2}?"},
#                         {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x2), conv(example_y2)))}"},
#                         {"role": "user", "content": f"What is the LCM of {x[n]} and {y[n]}?"},
#                     ]

#         elif problem_type == "square_mod":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]}^2 mod {y[n]}?"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1}^2 mod {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv((conv(example_x1))**2 % conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2}^2 mod {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv((conv(example_x2))**2 % conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]}^2 mod {y[n]}?"},
#                 ]

#         elif problem_type == "bitwise_and":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} AND {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} AND {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1) & conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} AND {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2) & conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]} AND {y[n]}?"},
#                 ]

#         elif problem_type == "bitwise_xor":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} XOR {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} XOR {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1) ^ conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} XOR {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2) ^ conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]} XOR {y[n]}?"},
#                 ]
#         elif problem_type == "bitwise_or":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} OR {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} OR {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x1) | conv(example_y1))}"},
#                     {"role": "user", "content": f"What is {example_x2} OR {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(conv(example_x2) | conv(example_y2))}"},
#                     {"role": "user", "content": f"What is {x[n]} OR {y[n]}?"},
#                 ]
#         elif problem_type == "bitwise_nor":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} NOR {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} NOR {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) | conv(example_y1)))}"},
#                     {"role": "user", "content": f"What is {example_x2} NOR {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) | conv(example_y2)))}"},
#                     {"role": "user", "content": f"What is {x[n]} NOR {y[n]}?"},
#                 ]
#         elif problem_type == "bitwise_nand":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} NAND {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} NAND {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) & conv(example_y1)))}"},
#                     {"role": "user", "content": f"What is {example_x2} NAND {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) & conv(example_y2)))}"},
#                     {"role": "user", "content": f"What is {x[n]} NAND {y[n]}?"},
#                 ]
#         elif problem_type == "bitwise_nxor":
#             if cot:
#                 dialog[n] += [
#                     {"role": "user", "content": f"Solve the following problem step by step: " 
#                      f"What is {x[n]} NXOR {y[n]}? Remember to convert your final answer back to decimal"},
#                 ]
#             else:
#                 dialog[n] += [
#                     {"role": "user", "content": f"What is {example_x1} NXOR {example_y1}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) ^ conv(example_y1)))}"},
#                     {"role": "user", "content": f"What is {example_x2} NXOR {example_y2}?"},
#                     {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) ^ conv(example_y2)))}"},
#                     {"role": "user", "content": f"What is {x[n]} NXOR {y[n]}?"},
#                 ]


#     return [dialog, x, y, problem_type]


# ──────────────────────────── TEMPLATE POOL ────────────────────────────
_FMT_POOL = {
    # ───────────────── ADDITION ─────────────────
    "addition": [
        "{x} + {y}",
        "{x}+{y}",
        "Add {x} and {y}.",
        "Work out {x} + {y}.",
        "Sum {x} and {y}.",
        "Total of {x} and {y}.",
        "Add together {x} and {y}.",
        "What is {x} plus {y}?",
        "What is the sum of {x} and {y}?",
        "Calculate {x} + {y}.",
    ],
    # ───────────────── MULTIPLICATION ─────────────────
    "multiplication": [
        "{x}*{y}{m}",
        "{x} * {y}{m}",
        "Calculate {x}*{y}{m}.",
        "Calculate {x} * {y}{m}.",
        "Work out {x}*{y}{m}.",
        "Work out {x} * {y}{m}.",
        "Work out {x} times {y}{m}.",
        "Multiply {x} and {y}{m}.",
        "Product of {x} and {y}{m}.",
        "What is the product of {x} and {y}{m}?",
        "{x} times {y}{m}",
        "What is {x} times {y}{m}?",
        "Find the result of {x} multiplied by {y}{m}.",
    ],
    # ───────────────── DIVISION ─────────────────
    "division": [
        "{x}//{y}",
        "{x} // {y}",
        "Divide {x} by {y}.",
        "{x} divided by {y}",
        "What is {x} divided by {y}?",
        "Calculate {x} divided by {y}.",
        "Compute {x} over {y}.",
    ],
    # ───────────────── MODULO ─────────────────
    "modulo": [
        "{x} mod {y}",
        "{x}%{y}",
        "{x} % {y}",
        "Find {x} mod {y}.",
        "What is {x} mod {y}?",
        "Calculate {x} modulo {y}.",
        "Compute {x} mod {y}.",
    ],
    # ───────────────── GCD ─────────────────
    "gcd": [
        "gcd({x}, {y})",
        "GCD({x}, {y})",
        "What is the GCD of {x} and {y}?",
        "Calculate the greatest common divisor of {x} and {y}.",
        "Find gcd of {x} and {y}.",
        "Compute GCD({x}, {y}).",
    ],
    # ───────────────── LCM ─────────────────
    "lcm": [
        "Find lcm({x}, {y}){m}.",
        "What is the least common multiple of {x} and {y}{m}?",
        "Calculate LCM({x}, {y}){m}.",
        "LCM({x}, {y}){m}",
        "Compute the least common multiple of {x} and {y}{m}.",
    ],
    # ───────────────── SQUARE MOD ─────────────────
    "square_mod": [
        "{x}^2 mod {y}",
        "({x}^2) mod {y}",
        "What is {x} squared mod {y}?",
        "Calculate {x}^2 mod {y}.",
        "Compute {x} squared modulo {y}.",
    ],
    # ───────────────── BITWISE AND ─────────────────
    "bitwise_and": [
        "{x} & {y}",
        "{x}&{y}",
        "{x} AND {y}",
        "Calculate {x} AND {y}.",
        "Compute the bitwise AND of {x} and {y}.",
        "What is {x} AND {y}?",
    ],
    # ───────────────── BITWISE OR ─────────────────
    "bitwise_or": [
        "{x} | {y}",
        "{x}|{y}",
        "{x} OR {y}",
        "Calculate {x} OR {y}.",
        "Compute the bitwise OR of {x} and {y}.",
        "What is {x} OR {y}?",
    ],
    # ───────────────── BITWISE XOR ─────────────────
    "bitwise_xor": [
        "{x} ^ {y}",
        "{x}^{y}",
        "{x} XOR {y}",
        "Calculate {x} XOR {y}.",
        "Compute the bitwise XOR of {x} and {y}.",
        "What is {x} XOR {y}?",
    ],
    # ───────────────── BITWISE NOR ─────────────────
    "bitwise_nor": [
        "{x} NOR {y}",
        "Bitwise NOR of {x} and {y}.",
        "Calculate {x} NOR {y}.",
        "Compute the bitwise NOR of {x} and {y}.",
        "What is {x} NOR {y}?",
    ],
    # ───────────────── BITWISE NAND ─────────────────
    "bitwise_nand": [
        "{x} NAND {y}",
        "Bitwise NAND of {x} and {y}.",
        "Calculate {x} NAND {y}.",
        "Compute the bitwise NAND of {x} and {y}.",
        "What is {x} NAND {y}?",
    ],
    # ───────────────── BITWISE NXOR ─────────────────
    "bitwise_nxor": [
        "{x} NXOR {y}",
        "{x} XNOR {y}",
        "Bitwise NXOR of {x} and {y}.",
        "Calculate {x} NXOR {y}.",
        "Compute the bitwise NXOR of {x} and {y}.",
        "What is {x} NXOR {y}?",
    ],
}

# ──────────────────────────── HELPERS ────────────────────────────────
def _rand_template(
    problem_type: str,
    x_val: int | str,
    y_val: int | str,
    limit_solution_digits: bool,
    complexity: int,
) -> str:
    """
    Draw a random template for ``problem_type`` from ``_FMT_POOL``
    and fill in x, y (and m if relevant).

    Parameters
    ----------
    x_val, y_val
        Already in the desired representation (int or word form).
    """
    tmpl = random.choice(_FMT_POOL[problem_type])
    if "{m}" in tmpl:
        m = f" mod {10 ** (complexity + 1)}" if limit_solution_digits else ""
        return tmpl.format(x=x_val, y=y_val, m=m)
    return tmpl.format(x=x_val, y=y_val)


# ────────────────────────── MAIN FUNCTION ────────────────────────────
def generate_dialog(
    complexity: int = 8,
    samples: int = 1,
    problem_type: str | list[str] = "addition",
    cot: bool = False,
    string_nums: bool = False,
    limit_solution_digits: bool = True,
    modify_question_format: bool = False,
) -> Tuple[List[List[dict]], np.ndarray, np.ndarray, str]:
    """
    Generate a list of ChatML dialogs for basic arithmetic / bitwise tasks.
    The public signature is unchanged – only the question-format logic has
    been centralised so *all* problem types benefit from the template pool
    when ``modify_question_format=True``.
    """
    rng = np.random.default_rng()

    # ── sample operands ───────────────────────────
    x = rng.integers(1, 10 ** (complexity + 1), size=samples)
    y = rng.integers(1, 10 ** (complexity + 1), size=samples)

    # ensure x ≥ y for a nicer canonical ordering
    for i in range(samples):
        x[i], y[i] = max(x[i], y[i]), min(x[i], y[i])

    # word-form conversion if requested
    if string_nums:
        x_words = np.array([n2w.num2words(num) for num in x])
        y_words = np.array([n2w.num2words(num) for num in y])
        x, y, x_words, y_words = x_words, y_words, x, y  # stash ints

    # example pairs for few-shot prefix (unchanged logic)
    ex1, ex2 = rng.integers(1, 10 ** (complexity + 1), size=(2, 2))
    ex1 = tuple(sorted(ex1, reverse=True))
    ex2 = tuple(sorted(ex2, reverse=True))
    if string_nums:
        ex1_words = tuple(n2w.num2words(num) for num in ex1)
        ex2_words = tuple(n2w.num2words(num) for num in ex2)

    # helpers for numeric / word conversion
    if string_nums:
        conv = lambda z: w2n.word_to_num(str(z))
        conv_inv = lambda z: n2w.num2words(int(z))
    else:
        conv = conv_inv = lambda z: z

    # ── select / randomise problem type ───────────
    if isinstance(problem_type, list):
        problem_type = random.choice(problem_type)
    if problem_type == "random":
        problem_type = random.choice(list(_FMT_POOL.keys()))

    dialog: List[List[dict]] = []

    for i in range(samples):
        # initial system prompt
        dialog.append(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a math-solving assistant. Always explain your reasoning step by step. "
                        "Regardless of the steps taken, ensure the final answer is clearly marked with 'Final Answer: x'."
                    )
                    if cot
                    else (
                        "You are a math solving helper. Don't use any commas in your output, "
                        "and always answer problems according to the format of previous answers."
                    )
                }
            ]
        )

        # convenience handles for current operands
        x_curr, y_curr = x[i], y[i]

        # function-local shortcuts for templates
        ask = lambda xx, yy: {"role": "user", "content": _rand_template(problem_type, xx, yy, limit_solution_digits, complexity)}

        # ╭────────────────────────── PROBLEM-TYPE SWITCH ─────────────────────────╮
        if problem_type == "addition":
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is {x_curr} plus {y_curr}?",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is {ex1[0]} plus {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex1[0]) + conv(ex1[1]))}"},
                        {"role": "user", "content": f"What is {ex2[0]} plus {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex2[0]) + conv(ex2[1]))}"},
                    ]
                )
                if modify_question_format:
                    dialog[i].append(ask(x_curr, y_curr))
                else:
                    dialog[i].append({"role": "user", "content": f"What is {x_curr} plus {y_curr}?"})

        elif problem_type == "multiplication":
            mod_term = f" mod {10 ** (complexity + 1)}" if limit_solution_digits else ""
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": (
                            f"Solve the following problem step by step: "
                            f"What is {x_curr} times {y_curr}{mod_term}?"
                        ),
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {
                            "role": "user",
                            "content": f"What is {ex1[0]} times {ex1[1]}{mod_term}?",
                        },
                        {
                            "role": "assistant",
                            "content": f"{conv_inv((conv(ex1[0]) * conv(ex1[1])) % 10 ** (complexity + 1) if limit_solution_digits else conv(ex1[0]) * conv(ex1[1]))}",
                        },
                        {
                            "role": "user",
                            "content": f"What is {ex2[0]} times {ex2[1]}{mod_term}?",
                        },
                        {
                            "role": "assistant",
                            "content": f"{conv_inv((conv(ex2[0]) * conv(ex2[1])) % 10 ** (complexity + 1) if limit_solution_digits else conv(ex2[0]) * conv(ex2[1]))}",
                        },
                    ]
                )
                if modify_question_format:
                    dialog[i].append(ask(x_curr, y_curr))
                else:
                    dialog[i].append(
                        {
                            "role": "user",
                            "content": f"What is {x_curr} times {y_curr}{mod_term}?",
                        }
                    )

        # ───────────── Remaining problem types ─────────────
        # Division
        elif problem_type == "division":
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is {x_curr} // {y_curr}?",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is {ex1[0]} // {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex1[0]) // conv(ex1[1]))}"},
                        {"role": "user", "content": f"What is {ex2[0]} // {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex2[0]) // conv(ex2[1]))}"},
                    ]
                )
                dialog[i].append(ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is {x_curr} // {y_curr}?"})

        # Modulo
        elif problem_type == "modulo":
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is {x_curr} mod {y_curr}?",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is {ex1[0]} mod {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex1[0]) % conv(ex1[1]))}"},
                        {"role": "user", "content": f"What is {ex2[0]} mod {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(conv(ex2[0]) % conv(ex2[1]))}"},
                    ]
                )
                dialog[i].append(ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is {x_curr} mod {y_curr}?"})

        # GCD
        elif problem_type == "gcd":
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is the GCD of {x_curr} and {y_curr}?",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is the GCD of {ex1[0]} and {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(ex1[0]), conv(ex1[1])))}"},
                        {"role": "user", "content": f"What is the GCD of {ex2[0]} and {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(ex2[0]), conv(ex2[1])))}"},
                    ]
                )
                dialog[i].append(ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is the GCD of {x_curr} and {y_curr}?"})

        # LCM
        elif problem_type == "lcm":
            mod_term = f" mod {10 ** (complexity + 1)}" if limit_solution_digits else ""
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is the LCM of {x_curr} and {y_curr}{mod_term}?",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is the LCM of {ex1[0]} and {ex1[1]}{mod_term}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(ex1[0]), conv(ex1[1])) % 10 ** (complexity + 1) if limit_solution_digits else np.lcm(conv(ex1[0]), conv(ex1[1])))}"},
                        {"role": "user", "content": f"What is the LCM of {ex2[0]} and {ex2[1]}{mod_term}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(ex2[0]), conv(ex2[1])) % 10 ** (complexity + 1) if limit_solution_digits else np.lcm(conv(ex2[0]), conv(ex2[1])))}"},
                    ]
                )
                dialog[i].append(ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is the LCM of {x_curr} and {y_curr}{mod_term}?"})

        # Square mod
        elif problem_type == "square_mod":
            # identical for cot vs non-cot except few-shot examples
            user_q = ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is {x_curr}^2 mod {y_curr}?"}
            if cot:
                dialog[i].append({"role": "user", "content": f"Solve the following problem step by step: What is {x_curr}^2 mod {y_curr}?"})
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is {ex1[0]}^2 mod {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(ex1[0]) ** 2) % conv(ex1[1]))}"},
                        {"role": "user", "content": f"What is {ex2[0]}^2 mod {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(ex2[0]) ** 2) % conv(ex2[1]))}"},
                    ]
                )
                dialog[i].append(user_q)

        # Bitwise family
        else:
            op_map = {
                "bitwise_and": ("AND", lambda a, b: a & b),
                "bitwise_or": ("OR", lambda a, b: a | b),
                "bitwise_xor": ("XOR", lambda a, b: a ^ b),
                "bitwise_nor": ("NOR", lambda a, b: ~(a | b)),
                "bitwise_nand": ("NAND", lambda a, b: ~(a & b)),
                "bitwise_nxor": ("NXOR", lambda a, b: ~(a ^ b)),
            }
            op_str, op_fn = op_map[problem_type]
            if cot:
                dialog[i].append(
                    {
                        "role": "user",
                        "content": f"Solve the following problem step by step: What is {x_curr} {op_str} {y_curr}? Remember to convert your final answer back to decimal",
                    }
                )
            else:
                dialog[i].extend(
                    [
                        {"role": "user", "content": f"What is {ex1[0]} {op_str} {ex1[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(op_fn(conv(ex1[0]), conv(ex1[1])))}"},
                        {"role": "user", "content": f"What is {ex2[0]} {op_str} {ex2[1]}?"},
                        {"role": "assistant", "content": f"{conv_inv(op_fn(conv(ex2[0]), conv(ex2[1])))}"},
                    ]
                )
                dialog[i].append(ask(x_curr, y_curr) if modify_question_format else {"role": "user", "content": f"What is {x_curr} {op_str} {y_curr}?"})

    return dialog, x, y, problem_type


def generate_non_math_dialog(samples=1, topic="philosophy", cot=False):
    example_questions = {
        "philosophy": [
            ("Is the Ship of Theseus still the same ship after all parts are replaced?", "This is a classic thought experiment questioning the nature of identity."),
            ("What is the meaning of life according to existentialism?", "Existentialists argue that individuals create their own meaning through choices and actions."),
            ("Can free will exist in a deterministic universe?", "This question explores the compatibility of determinism with the concept of free will."),
            ("What distinguishes knowledge from belief?", "Knowledge typically requires justified true belief, while belief does not necessarily require justification."),
            ("Does morality exist independently of humans?", "This question examines moral realism versus moral anti-realism.")
        ],
        "ethics": [
            ("Is it morally acceptable to lie to protect someone's feelings?", "This involves balancing honesty with the value of kindness."),
            ("Should animals have the same rights as humans?", "This raises questions about sentience, suffering, and ethical consideration."),
            ("Is it ethical to use artificial intelligence in decision-making?", "This question considers fairness, accountability, and potential biases in AI systems."),
            ("Does the end justify the means?", "This touches on consequentialist versus deontological ethical theories."),
            ("Is capital punishment morally justifiable?", "This question explores justice, deterrence, and the value of human life.")
        ],
        "history": [
            ("What if the Roman Empire never fell?", "Speculative history suggests it could have led to advanced technology earlier."),
            ("How did the Industrial Revolution change society?", "It shifted economies from agrarian to industrial, reshaping labor and urbanization."),
            ("What were the causes and consequences of the French Revolution?", "It led to the rise of democracy and the decline of monarchies in Europe."),
            ("How did the Cold War influence global politics?", "It created a bipolar world order, leading to numerous proxy wars and political tensions."),
            ("What if World War II had a different outcome?", "This explores alternate history scenarios with potential geopolitical shifts.")
        ],
        "psychology": [
            ("What does the Stanford Prison Experiment reveal about human behavior?", "It highlights the power of situational influences over personal traits."),
            ("How do cognitive biases affect decision-making?", "Biases like confirmation bias can distort our perception and judgments."),
            ("What is the impact of social media on mental health?", "It can influence self-esteem, anxiety levels, and social connections both positively and negatively."),
            ("How does memory work in the human brain?", "Memory involves encoding, storage, and retrieval processes within neural networks."),
            ("What role does nature versus nurture play in personality development?", "This explores the influence of genetics and environment on behavior.")
        ],
        "science_fiction": [
            ("What are the ethical implications of artificial intelligence surpassing human intelligence?", "This involves concerns about autonomy, control, and societal impact."),
            ("Could time travel ever be possible according to current physics?", "While speculative, theories like wormholes explore this possibility."),
            ("How might colonizing Mars change human society?", "It could lead to new cultural developments, governance systems, and ethical dilemmas."),
            ("What are the potential risks of genetic engineering?", "Concerns include unintended consequences, ethical issues, and impacts on biodiversity."),
            ("What would society look like in a post-scarcity economy?", "It would challenge traditional economic models and social structures.")
        ],
        "technology": [
            ("How has the internet changed the way we communicate?", "It has enabled instant, global communication but also introduced challenges like misinformation."),
            ("What are the ethical concerns with facial recognition technology?", "Issues include privacy invasion, surveillance, and potential biases."),
            ("Will quantum computing revolutionize cybersecurity?", "Quantum computing poses both opportunities and risks for data encryption and security."),
            ("How does blockchain technology work?", "It is a decentralized, secure method for recording transactions using cryptographic techniques."),
            ("What is the future of autonomous vehicles?", "Advancements may lead to changes in transportation, safety, and urban planning.")
        ],
        "art_and_culture": [
            ("What defines a work of art?", "Art can be defined by its aesthetic value, emotional impact, or cultural significance."),
            ("How has pop culture influenced societal norms?", "Pop culture reflects and shapes attitudes, trends, and behaviors in society."),
            ("What role does art play in social movements?", "Art can inspire, provoke thought, and mobilize people for causes."),
            ("How does music affect human emotions?", "Music influences mood, cognitive functions, and even physiological responses."),
            ("What is the significance of cultural heritage?", "It preserves the identity, history, and values of communities across generations.")
        ]
    }

    dialog: List[Dialog] = []
    correct_responses    = []
    

    if topic == "random":
        topic = random.choice(list(example_questions.keys()))

    if isinstance(topic, list):
        topic = random.choice(topic)

    for _ in range(samples):
        example_1, response_1 = random.choice(example_questions[topic])
        example_2, response_2 = random.choice(example_questions[topic])
        new_question, new_response = random.choice(example_questions[topic])
        correct_responses += [new_response]

        if cot:
            dialog.append([
                {"role": "system", "content": "You are a thoughtful assistant. Provide reasoned and reflective answers."},
                {"role": "user", "content": f"Consider this question carefully: {new_question}"},
            ])
        else:
            dialog.append([
                {"role": "system", "content": "You are a knowledgeable assistant providing concise answers."},
                # Don't need multi-shot prompting, since this is a general knowledge question with no specific output format
                #{"role": "user", "content": f"{example_1}"},
                #{"role": "assistant", "content": f"{response_1}"},
                #{"role": "user", "content": f"{example_2}"},
                #{"role": "assistant", "content": f"{response_2}"},
                {"role": "user", "content": f"{new_question}"}
            ])

    return [dialog, correct_responses, topic]

def episode(generator, dialogs, temperature=0.0, top_p=0.9, inference_mode=None,
            max_decoding_length=100, curr_pt="addition", curr_x=0, curr_y=0, verbose=False):
    
    if type(inference_mode) == type(None):
        inference_mode = generator.model.forward

    prompt_tokens = generator.parse_chat(dialogs)

    max_gen_len = generator.model.params.max_seq_len - 1
    top_p = top_p
    echo = False

    params = generator.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = generator.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz)
    input_text_mask = tokens != pad_id

    stop_tokens = torch.tensor(list(generator.tokenizer.stop_tokens))

    transitions = []
    curr_token = 0
    list_of_probs  = []
    list_of_logits = []
    h_stacks = []
    for cur_pos in range(min_prompt_len, total_len):
        logits, h_stack, h = inference_mode(tokens[:, prev_pos:cur_pos], prev_pos, curr_token=curr_token, curr_pt=curr_pt, curr_x=curr_x, curr_y=curr_y, verbose=verbose)
        # Shape of logits are (batch_size, total_sequence_length, num_tokens)
        h_stacks += [h_stack]
        # probs are intentionally being calculated here, so that it contains an extra token (the stop token), to help with loss calculation
        probs = torch.softmax(logits[:, -1,:] / 1, dim=-1)
        list_of_probs  += [probs]
        list_of_logits += [logits[:,-1,:]]
        new_logits = logits
        if temperature > 0:
            probs = torch.softmax(new_logits[:, -1] / temperature, dim=-1)
            #print(logits, logits.shape)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(new_logits[:, -1], dim=-1)
        if curr_token > max_decoding_length:
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

        curr_token += 1
        
    list_of_probs = torch.stack(list_of_probs)
    list_of_logits = torch.stack(list_of_logits)

    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        # cut to after eos tok if any
        for stop_token in generator.tokenizer.stop_tokens:
            try:
                eos_idx = toks.index(stop_token)
                toks = toks[:eos_idx]
            except ValueError:
                pass
        out_tokens.append(toks)

    return h_stacks, list_of_probs, list_of_logits, out_tokens


def gather_h_stacks(generator, SE, dialog_data, temperature=0, produce_correct_VSA=False):
    dialogs = dialog_data[0]

    h_stacks, list_of_probs, list_of_logits, out_tokens = episode(generator, dialogs, temperature=temperature,
                                                                  inference_mode=generator.model.forward, 
                                                                  max_decoding_length=1,
                                                                  )
    
    # shape of h_stack is [num_layers, batch_size, num_tokens, hidden_dm], per output token
    

    if produce_correct_VSA:
        x       = dialog_data[1]
        y       = dialog_data[2]
        problem_type = dialog_data[3]
        correct_VSAs = []
        for n in range(len(x)):
            correct_VSA   = SE.generate_VSA_old(x[n], y[n], problem_type).to(torch.bfloat16)
            correct_VSAs += [correct_VSA.flatten()]
        correct_VSAs = torch.stack(correct_VSAs)

        # return h_stack[0], since we are not concerned with the LLM output at this stage
        return h_stacks[0], correct_VSAs
    else:
        return h_stacks[0], None



def get_dialog_indices(generator, dialog, calculate_end_index=False):
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


def generate_and_save_data(generator, SE, save_dir, rounds, mode, save_frequency, complexity, n_samples, problem_type, df_subset=None,
                           tokens_to_keep=1, calculate_end_index=False, verbose=True):

    if type(df_subset) == type(None):
        use_existing_questions = False
    else:
        use_existing_questions = True

    if mode == "train":
        h_path  = 'h_stack_round_'
        sp_path = 'correct_sps_round_'
    elif mode == "val":
        h_path  = 'validation_h_stack_round_'
        sp_path = 'validation_correct_sps_round_'
    elif mode == "test":
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
                h_stacked = h_stacked.view(-1, tokens_to_keep, generator.model.params.dim, generator.model.params.n_layers+1)
            # When saved, the shape of h_stacked is (batch, num_tokens, hidden_dim, n_layers)
            torch.save(h_stacked, os.path.join(save_dir, f"{h_path}{r}.pt"))
            torch.save(numbers_stacked, os.path.join(save_dir, f"{sp_path}{r}.pt"))
            h_stacks = []
            numbers  = []
            if r == rounds:
                break

        if use_existing_questions:
            batch = df_subset.iloc[r * n_samples : (r + 1) * n_samples]
            question, problem_type = batch["question"], batch["problem_type"]
            x, y, solution         = batch["x"], batch["y"], batch["solution"]

            # batch_dialog_data is a list of lists, with length equal to the length of df_dialogs. Each list contains 4 items. The first is 
            #  the dialogs object, which is a list of Dialog objects, the length of which is equal to n_samples. The second is the x values, which is 
            #  an array of integers, the third is the y values (also array of integers), and the final is the problem type, a string
            dialog_data = [generate_dialog(complexity=complexity, samples=1,
                                        problem_type=pt) for pt in problem_type]

            for d in range(n_samples):
                # First index is grabbing the batch item, second index is grabbing the dialog (instead of the x, y , pt),
                #  third index is grabbing the batch item within dialogs (which is always of length 1 due to samples=1 above), and last
                #  index is grabbing the last dialog sequence, since we only want to change that while leaving the example dialogs the same
                dialog_data[d][0][0][-1]['content'] = question.values[d]
                dialog_data[d][1][0], dialog_data[d][2][0] = x.values[d], y.values[d]


            # dialog_data should be [dialog, x, y, pt], where each element is n_samples long. dialog_data previously was of length n_samples, where each
            #  item in the sequence was [dialog, x, y, pt]. The below code puts it into the correct format
            dialog_data = [[d[0][0] for d in dialog_data],
                           np.array([d[1][0] for d in dialog_data]),
                           np.array([d[2][0] for d in dialog_data]),
                           problem_type.values]

            correct_vsas = SE.generate_VSA(torch.tensor(dialog_data[1]), torch.tensor(dialog_data[2]), dialog_data[3]).type(torch.bfloat16)

            append_dialog_batch_to_file(dialog_data[0], "dialog_data_log_pregen.jsonl")
            h_stack, _ = gather_h_stacks(generator, SE, dialog_data, produce_correct_VSA=False)
        else:
            # Generate dialog data and gather 'h_stack' and 'correct_sps'
            dialog_data = generate_dialog(complexity=complexity, samples=n_samples, problem_type=problem_type)
            append_dialog_batch_to_file(dialog_data[0], "dialog_data_log_random.jsonl")

            h_stack, correct_vsas = gather_h_stacks(generator, SE, dialog_data, produce_correct_VSA=True)

        # shape of h_stack is n_layers, batch, num_tokens, hiddem_dim.

        if tokens_to_keep == "all":
            # Dialog_data[0] is the dialogs 
            start_indices, end_indices = get_dialog_indices(generator, dialog_data[0], calculate_end_index=calculate_end_index)
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
        numbers += correct_vsas


def generate_data_loaders(mode, save_dir, data_rounds, save_frequency, layer_numbers, n_samples=1, df_subset=None,
                          restrict_dataset=None, tokens_to_keep=1, batch_size=512, gpu_seed=False, verbose=False):
    if mode == "train":
        h_path  = 'h_stack_round_'
        sp_path = 'correct_sps_round_'
        shuffle = True
    elif mode == "val":
        h_path  = 'validation_h_stack_round_'
        sp_path = 'validation_correct_sps_round_'
        shuffle = False
    elif mode == "test":
        h_path  = 'testing_h_stack_round_'
        sp_path = 'testing_correct_sps_round_'
        shuffle = False

    # Load data for each layer to create data loaders
    encoder_data_loaders = []

    for n_layer in layer_numbers:
        if verbose:
            print("--- On Layer Number:", n_layer.item())
        h_layer_data = []
        correct_sps_data = []

        # Load each round's data from disk
        if restrict_dataset: # If restrict_dataset is not set to None or 0, then reduce the amount of runs loaded to restrict_dataset
            runs = restrict_dataset
        else:
            runs = data_rounds
        for r in range(runs+1):
            if not r % save_frequency and r:
                if verbose:
                    print("On Round Number:", r)
                h_stack = torch.load(os.path.join(save_dir, f"{h_path}{r}.pt"), weights_only=True)
                correct_sps = torch.load(os.path.join(save_dir, f"{sp_path}{r}.pt"), weights_only=True)

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
        if gpu_seed:
            gpu_generator.manual_seed(42)

        encoder_data_loader = DataLoader(
            encoder_training_data,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=gpu_generator,
        )
        encoder_data_loaders.append(encoder_data_loader)
        
    return encoder_data_loaders

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

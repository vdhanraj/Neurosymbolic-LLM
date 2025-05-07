
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


def generate_dialog(complexity=8, # complexity + 1 is the maximum number of digits the generated numbers (input and output) can be
                    samples=1, # Number of samples to generate. Note all generated samples will be of the same problem type
                    problem_type="addition", # Problem type to generate. If set to "random" or a list of problem types, randomly select a problem type from either the entire set of possible problems or the specified subset, repsectively
                    cot=False,  # If true, use CoT prompting
                    string_nums=False, # If true, represent numbers as words (e.g., two hundred and one)
                    limit_solution_digits=True, # If True, certain problem types whose solutions have more digits than their inputs will have their solutions truncated (via solution mod 10^(complexity + 1))
                    modify_question_format=False, # If True, modify the manner in which questions are asked (e.g., ask "What is 12 * 32" or "Multiply 12 and 32" instead of "What is 12 times 32")
                    ):
    #x = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
    x = np.random.randint(low=1, high=10**(complexity+1), size=samples)
    #y = np.random.randint(low=10**(complexity), high=10**(complexity+1), size=samples)
    y = np.random.randint(low=1, high=10**(complexity+1), size=samples)
    
    temp_x = []
    temp_y = []
    for n in range(samples):
        x[n], y[n] = max(x[n], y[n]), min(x[n], y[n])
        if string_nums:
            temp_x += [n2w.num2words(x[n])]
            temp_y += [n2w.num2words(y[n])]
    if string_nums:
        x, y, temp_x, temp_y = np.array(temp_x), np.array(temp_y), x, y

    #example_x1, example_y1 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
    example_x1, example_y1 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
    #example_x2, example_y2 = np.random.randint(low=10**(complexity), high=10**(complexity+1)), np.random.randint(low=10**(complexity), high=10**(complexity+1))
    example_x2, example_y2 = np.random.randint(low=1, high=10**(complexity+1)), np.random.randint(low=1, high=10**(complexity+1))
    example_x1, example_y1 = max(example_x1, example_y1), min(example_x1, example_y1)
    example_x2, example_y2 = max(example_x2, example_y2), min(example_x2, example_y2)
    
    if string_nums:
        example_x1, example_y1 = n2w.num2words(example_x1), n2w.num2words(example_y1)
        example_x2, example_y2 = n2w.num2words(example_x2), n2w.num2words(example_y2)

    if string_nums:
        conv = lambda x: w2n.word_to_num(str(x))
        conv_inv = lambda x: n2w.num2words(int(x))

    else:
        conv = lambda x: x
        conv_inv = lambda x: x


    dialog: List[Dialog] = []

    if type(problem_type) == type([]):
        problem_type = random.choice(problem_type)
    
    if problem_type == "random":
        problem_type = random.choice(["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"])

    for n in range(samples):
        if cot:
            dialog += [
                [
                    {"role": "system", "content": 
                     "You are a math-solving assistant. Always explain your reasoning step by step. "
                     "Regardless of the steps taken, ensure the final answer is clearly marked with 'Final Answer: x'."
                    },
                ]
            ]
        else:
            dialog += [
                [
                    {"role": "system", "content": 
                     "You are a math solving helper. Don't use any commas in your output, "
                     "and always answer problems according to the format of previous answers."
                    },
                ]
            ]

        if problem_type == "addition":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} plus {y[n]}?"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} plus {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1) + conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} plus {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2) + conv(example_y2))}"},
                ]

                if modify_question_format:
                    # Define the list of different formats
                    formats = [
                        "{x} + {y}",
                        "Work out {x} + {y}.",
                        "Calculate {x} + {y}.",
                        "What is {x} plus {y}?",
                        "Add {x} and {y}.",
                        "Sum of {x} and {y}.",
                        "What is the sum of {x} and {y}?",
                    ]
                    # Randomly pick one format
                    chosen_format = random.choice(formats)
                    # Format the question
                    question = chosen_format.format(x=x[n], y=y[n])
                    dialog[n] += [{"role": "user", "content": question}]
                else:
                    dialog[n] += [{"role": "user", "content": f"What is {x[n]} plus {y[n]}?"},]
        elif problem_type == "multiplication":
            if cot:
                if limit_solution_digits:
                    dialog[n] += [
                        {"role": "user", "content": f"Solve the following problem step by step: " 
                        f"What is {x[n]} times {y[n]} mod {10**(complexity+1)}?"},
                    ]
                else:
                    dialog[n] += [
                        {"role": "user", "content": f"Solve the following problem step by step: " 
                        f"What is {x[n]} times {y[n]}?"},
                    ]
            else:
                if limit_solution_digits:
                    dialog[n] += [
                        {"role": "user", "content": f"What is {example_x1} times {example_y1} mod {10**(complexity+1)}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(example_x1) * conv(example_y1)) % 10**(complexity+1))}"},
                        {"role": "user", "content": f"What is {example_x2} times {example_y2} mod {10**(complexity+1)}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(example_x2) * conv(example_y2)) % 10**(complexity+1))}"},
                    ]

                    if modify_question_format:
                        # For modular multiplication, only slight variations make sense
                        formats = [
                            "What is {x} times {y} mod {mod}?",
                            "Calculate {x} * {y} modulo {mod}.",
                            "Work out {x} times {y} mod {mod}.",
                            "Find the result of {x} multiplied by {y} modulo {mod}.",
                        ]
                        chosen_format = random.choice(formats)
                        question = chosen_format.format(x=x[n], y=y[n], mod=10**(complexity+1))
                        dialog[n] += [{"role": "user", "content": question}]
                    else:
                        dialog[n] += [{"role": "user", "content": f"What is {x[n]} times {y[n]} mod {10**(complexity+1)}?"}]

                else:
                    dialog[n] += [
                        {"role": "user", "content": f"What is {example_x1} times {example_y1}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(example_x1) * conv(example_y1)))}"},
                        {"role": "user", "content": f"What is {example_x2} times {example_y2}?"},
                        {"role": "assistant", "content": f"{conv_inv((conv(example_x2) * conv(example_y2)))}"},
                    ]

                    if modify_question_format:
                        # For normal multiplication
                        formats = [
                            "{x} * {y}",
                            "Work out {x} * {y}.",
                            "Calculate {x} * {y}.",
                            "What is {x} times {y}?",
                            "Multiply {x} and {y}.",
                            "Product of {x} and {y}.",
                            "What is the product of {x} and {y}?",
                        ]
                        chosen_format = random.choice(formats)
                        question = chosen_format.format(x=x[n], y=y[n])
                        dialog[n] += [{"role": "user", "content": question}]
                    else:
                        dialog[n] += [{"role": "user", "content": f"What is {x[n]} times {y[n]}?"}]


        elif problem_type == "division":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} // {y[n]}?"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} // {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1)//conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} // {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2)//conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]} // {y[n]}?"},
                ]

        elif problem_type == "modulo":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} mod {y[n]}?"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} mod {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1) % conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} mod {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2) % conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]} mod {y[n]}?"},
                ]

        elif problem_type == "gcd":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is the GCD of {x[n]} and {y[n]}?"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is the GCD of {example_x1} and {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(example_x1), conv(example_y1)))}"},
                    {"role": "user", "content": f"What is the GCD of {example_x2} and {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(np.gcd(conv(example_x2), conv(example_y2)))}"},
                    {"role": "user", "content": f"What is the GCD of {x[n]} and {y[n]}?"},
                ]

        elif problem_type == "lcm":
            if cot:
                if limit_solution_digits:
                    dialog[n] += [
                        {"role": "user", "content": f"Solve the following problem step by step: " 
                        f"What is the LCM of {x[n]} and {y[n]} mod {10**(complexity+1)}?"},
                    ]
                else:
                    dialog[n] += [
                        {"role": "user", "content": f"Solve the following problem step by step: " 
                        f"What is the LCM of {x[n]} and {y[n]}?"},
                    ]

            else:
                if limit_solution_digits:
                    dialog[n] += [
                        {"role": "user", "content": f"What is the LCM of {example_x1} and {example_y1} mod {10**(complexity+1)}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x1), conv(example_y1)) % 10**(complexity+1))}"},
                        {"role": "user", "content": f"What is the LCM of {example_x2} and {example_y2} mod {10**(complexity+1)}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x2), conv(example_y2)) % 10**(complexity+1))}"},
                        {"role": "user", "content": f"What is the LCM of {x[n]} and {y[n]} mod {10**(complexity+1)}?"},
                    ]
                else:
                    dialog[n] += [
                        {"role": "user", "content": f"What is the LCM of {example_x1} and {example_y1}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x1), conv(example_y1)))}"},
                        {"role": "user", "content": f"What is the LCM of {example_x2} and {example_y2}?"},
                        {"role": "assistant", "content": f"{conv_inv(np.lcm(conv(example_x2), conv(example_y2)))}"},
                        {"role": "user", "content": f"What is the LCM of {x[n]} and {y[n]}?"},
                    ]

        elif problem_type == "square_mod":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]}^2 mod {y[n]}?"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1}^2 mod {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv((conv(example_x1))**2 % conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2}^2 mod {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv((conv(example_x2))**2 % conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]}^2 mod {y[n]}?"},
                ]

        elif problem_type == "bitwise_and":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} AND {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} AND {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1) & conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} AND {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2) & conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]} AND {y[n]}?"},
                ]

        elif problem_type == "bitwise_xor":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} XOR {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} XOR {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1) ^ conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} XOR {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2) ^ conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]} XOR {y[n]}?"},
                ]
        elif problem_type == "bitwise_or":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} OR {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} OR {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x1) | conv(example_y1))}"},
                    {"role": "user", "content": f"What is {example_x2} OR {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(conv(example_x2) | conv(example_y2))}"},
                    {"role": "user", "content": f"What is {x[n]} OR {y[n]}?"},
                ]
        elif problem_type == "bitwise_nor":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} NOR {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} NOR {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) | conv(example_y1)))}"},
                    {"role": "user", "content": f"What is {example_x2} NOR {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) | conv(example_y2)))}"},
                    {"role": "user", "content": f"What is {x[n]} NOR {y[n]}?"},
                ]
        elif problem_type == "bitwise_nand":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} NAND {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} NAND {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) & conv(example_y1)))}"},
                    {"role": "user", "content": f"What is {example_x2} NAND {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) & conv(example_y2)))}"},
                    {"role": "user", "content": f"What is {x[n]} NAND {y[n]}?"},
                ]
        elif problem_type == "bitwise_nxor":
            if cot:
                dialog[n] += [
                    {"role": "user", "content": f"Solve the following problem step by step: " 
                     f"What is {x[n]} NXOR {y[n]}? Remember to convert your final answer back to decimal"},
                ]
            else:
                dialog[n] += [
                    {"role": "user", "content": f"What is {example_x1} NXOR {example_y1}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x1) ^ conv(example_y1)))}"},
                    {"role": "user", "content": f"What is {example_x2} NXOR {example_y2}?"},
                    {"role": "assistant", "content": f"{conv_inv(~(conv(example_x2) ^ conv(example_y2)))}"},
                    {"role": "user", "content": f"What is {x[n]} NXOR {y[n]}?"},
                ]


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
                {"role": "user", "content": f"{example_1}"},
                {"role": "assistant", "content": f"{response_1}"},
                {"role": "user", "content": f"{example_2}"},
                {"role": "assistant", "content": f"{response_2}"},
                {"role": "user", "content": f"{new_question}"}
            ])

    return dialog, correct_responses, topic

def episode(generator, dialogs, temperature=0.0, top_p=0.9, inference_mode=None,
            max_decoding_length=100, verbose=False):
    
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
        logits, h_stack, h = inference_mode(tokens[:, prev_pos:cur_pos], prev_pos, curr_token=curr_token, verbose=verbose)
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


def gather_h_stacks(generator, SE, dialog_data, temperature=0):
    dialogs = dialog_data[0]
    x       = dialog_data[1]
    y       = dialog_data[2]
    problem_type = dialog_data[3]

    h_stacks, list_of_probs, list_of_logits, out_tokens = episode(generator, dialogs, temperature=temperature,
                                                                  inference_mode=generator.model.forward, 
                                                                  max_decoding_length=1,
                                                                  )
    
    # shape of h_stack is [num_layers, batch_size, num_tokens, hidden_dm], per output token
    
    correct_VSAs = []
    for n in range(len(x)):
        correct_VSA   = SE.generate_VSA(x[n], y[n], problem_type).to(torch.bfloat16)
        correct_VSAs += [correct_VSA.flatten()]
    correct_VSAs = torch.stack(correct_VSAs)

    # return h_stack[0], since we are not concerned with the LLM output at this stage
    return h_stacks[0], correct_VSAs



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


def generate_and_save_data(generator, SE, save_dir, rounds, mode, save_frequency, complexity, n_samples, problem_type, df_path="",
                           tokens_to_keep=1, calculate_end_index=False, verbose=True):

    if df_path == "":
        use_existing_questions = False
    else:
        df = pd.read_csv(df_path)
        use_existing_questions = True
        rounds = len(df) // n_samples
        if not len(df) % n_samples:
            print("Warning: size of predefined dataframe is not divisible by the LLM batch size. Some rows in the dataframe will not be processed")

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
            batch = df.iloc[r * n_samples : (r + 1) * n_samples]
            question, problem_type = batch["question"], batch["problem_type"]
            x, y, solution         = batch["x"], batch["y"], batch["solution"]

            dialog_data = [generate_dialog(complexity=complexity, samples=1,
                                        problem_type=pt) for pt in problem_type]
            for d in range(len(dialog_data)):
                dialog_data[d][0][0][-1]['content'] = question[d]
                dialog_data[d][1][0], dialog_data[d][2][0] = x[d], y[d]
        else:
            # Generate dialog data and gather 'h_stack' and 'correct_sps'
            dialog_data = generate_dialog(complexity=complexity, samples=n_samples, problem_type=problem_type)
        h_stack, correct_sps = gather_h_stacks(generator, SE, dialog_data)
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
        if restrict_dataset: # If restrict_dataset is not set to None, then reduce the amount of runs loaded to restrict_dataset
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
            generator=gpu_generator,
        )
        encoder_data_loaders.append(encoder_data_loader)
        
    return encoder_data_loaders

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

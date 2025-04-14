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
import yaml
from pathlib import Path
import datetime
import wandb

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import plotly.graph_objects as go

from typing import List, Optional
import fire

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from typing import List

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

######################################################

# Parser

pre_parser = argparse.ArgumentParser(description="Config Loader", add_help=False)
pre_parser.add_argument("--config", type=str, default="train_encoders_and_decoders_default_config.yaml", help="Path to YAML configuration file")
config_args, remaining_argv = pre_parser.parse_known_args()

with open(config_args.config, "r") as f:
    config_defaults = yaml.safe_load(f)


# --- Full CLI parser with YAML defaults ---
parser = argparse.ArgumentParser(description="Train Encoders and Decoders")
parser.add_argument("--config", type=str, default="train_encoders_and_decoders_default_config.yaml", help="Path to YAML configuration file")

# === Path config ===
parser.add_argument("--curr_dir", type=str, default=config_defaults.get("curr_dir"), help="Path to program root")
parser.add_argument("--git_dir", type=str, default=config_defaults.get("git_dir"), help="Path to project Git root")
parser.add_argument("--chpt_dir", type=str, default=config_defaults.get("chpt_dir"), help="Path to LLM checkpoint directory")
parser.add_argument("--tokenizer_path", type=str, default=config_defaults.get("tokenizer_path"), help="Path to tokenizer.model file")
parser.add_argument("--generate_data", type=int, default=config_defaults.get("generate_data"), help="Whether to generate training data")
parser.add_argument("--log_wandb", type=int, default=config_defaults.get("log_wandb"), help="Whether to log outputs to wandb")

# === Model config ===
parser.add_argument("--max_seq_len", type=int, default=config_defaults.get("max_seq_len"), help="Max sequence length for LLM")
parser.add_argument("--max_batch_size", type=int, default=config_defaults.get("max_batch_size"), help="Batch size for LLM generation")
parser.add_argument("--model_parallel_size", type=int, default=config_defaults.get("model_parallel_size"), help="Model parallelism size")
parser.add_argument("--model_dim", type=int, default=config_defaults.get("model_dim"), help="Model dimension (hidden size)")

# === Generation params ===
parser.add_argument("--top_p", type=float, default=config_defaults.get("top_p"), help="Nucleus sampling probability threshold")
parser.add_argument("--temperature", type=float, default=config_defaults.get("temperature"), help="Sampling temperature")
parser.add_argument("--max_gen_len", type=int, default=config_defaults.get("max_gen_len"), help="Maximum number of tokens to generate")

# === Symbolic Engine ===
parser.add_argument("--max_digits", type=int, default=config_defaults.get("max_digits"), help="Max number of digits to encode")
parser.add_argument("--VSA_dim", type=int, default=config_defaults.get("VSA_dim"), help="VSA dimensionality")
parser.add_argument("--possible_problems", nargs="+", type=str, default=config_defaults.get("possible_problems"), help="All problem types symbolic engine should support")

# === Data generation ===
parser.add_argument("--train_data_rounds", type=int, default=config_defaults.get("train_data_rounds"), help="Number of training queries to LLM")
parser.add_argument("--test_data_rounds", type=int, default=config_defaults.get("test_data_rounds"), help="Number of test queries to LLM")
parser.add_argument("--restrict_train_dataset", type=int, default=config_defaults.get("restrict_train_dataset"), help="Number of queries to load to train encoders and decoders")
parser.add_argument("--restrict_test_dataset", type=int, default=config_defaults.get("restrict_test_dataset"), help="Number of queries to load to test encoders and decoders")
parser.add_argument("--save_frequency", type=int, default=config_defaults.get("save_frequency"), help="Save hidden/VSA data every N batches")
parser.add_argument("--layer_numbers", type=int, nargs="+", default=config_defaults.get("layer_numbers"), help="Layers to train encoders/decoders for")
parser.add_argument("--complexity", type=int, default=config_defaults.get("complexity"), help="Problem complexity = digits + 1")
parser.add_argument("--n_samples", type=int, default=config_defaults.get("n_samples"), help="Number of samples to use per forward pass")
parser.add_argument("--problem_type", nargs="+", type=str, default=config_defaults.get("problem_type"), help="Subset of problem types to use for training")

# === Token handling ===
parser.add_argument("--tokens_to_keep", type=str, default=config_defaults.get("tokens_to_keep"), help="How many tokens to keep (or 'all')")
parser.add_argument("--calculate_end_index", type=bool, default=config_defaults.get("calculate_end_index"), help="Whether to cut tokens at the end of the prompt")

# === Encoder/Decoder training ===
parser.add_argument("--encoder_decoder_batch_size", type=int, default=config_defaults.get("encoder_decoder_batch_size"), help="Training batch size")
parser.add_argument("--training_epochs", type=int, default=config_defaults.get("training_epochs"), help="Number of epochs for encoder training")
parser.add_argument("--learning_rate", type=float, default=config_defaults.get("learning_rate"), help="Base learning rate for encoder")
parser.add_argument("--learning_rate_reduction_factors", type=json.loads, default=config_defaults.get("learning_rate_reduction_factors"), help="Epoch:Factor map for reducing LR")

parser.add_argument("--decoding_epochs", type=int, default=config_defaults.get("decoding_epochs"), help="Number of epochs for decoder training")
parser.add_argument("--decoding_learning_rate", type=float, default=config_defaults.get("decoding_learning_rate"), help="Base learning rate for decoder")
parser.add_argument("--decoding_learning_rate_reduction_factors", type=json.loads, default=config_defaults.get("decoding_learning_rate_reduction_factors"), help="Epoch:Factor map for reducing decoder LR")

args = parser.parse_args(remaining_argv)

# --- Step 5: Expand and assign ---
curr_dir       = str(Path(args.curr_dir).expanduser())
git_dir        = str(Path(args.git_dir).expanduser())
ckpt_dir       = str(Path(args.chpt_dir).expanduser())
tokenizer_path = str(Path(args.tokenizer_path).expanduser())
generate_data  = args.generate_data
log_wandb      = args.log_wandb

# Model + sampling
max_seq_len         = args.max_seq_len
max_batch_size      = args.max_batch_size
model_parallel_size = args.model_parallel_size
model_dim           = args.model_dim

top_p        = args.top_p
temperature  = args.temperature
max_gen_len  = args.max_gen_len

# Symbolic engine
max_digits         = args.max_digits
VSA_dim            = args.VSA_dim
possible_problems  = args.possible_problems

# Data collection
train_data_rounds      = args.train_data_rounds
test_data_rounds       = args.test_data_rounds
restrict_train_dataset = args.restrict_train_dataset
restrict_test_dataset  = args.restrict_test_dataset
save_frequency         = args.save_frequency
layer_numbers          = args.layer_numbers
complexity             = args.complexity
n_samples              = args.n_samples
problem_type           = args.problem_type

# Tokenization
tokens_to_keep      = args.tokens_to_keep
calculate_end_index = args.calculate_end_index

# Encoder/decoder training
encoder_decoder_training_batch_size      = args.encoder_decoder_batch_size
training_epochs                          = args.training_epochs
learning_rate                            = args.learning_rate
learning_rate_reduction_factors          = args.learning_rate_reduction_factors
decoding_training_epochs                 = args.decoding_epochs
decoding_learning_rate                   = args.decoding_learning_rate
decoding_learning_rate_reduction_factors = args.decoding_learning_rate_reduction_factors

######################################################

sys.path.insert(0, git_dir)

from llama.encoder_decoder_networks import Encoder, Decoder, Encoder_Deep, Decoder_Deep, LastTokenTransformer
from llama.vsa_engine import *
from llama.utilities import *

from llama import Llama

######################################################

curr_date = datetime.datetime.now().strftime("%Y%m%d")

if log_wandb:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", torch.cuda.get_device_name(torch.cuda.current_device()))

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
else:
    if torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)

    torch.set_default_device("cuda")


max_digits = 5 #15 # maximum representable number is 10**max_digits
VSA_dim = 2048

possible_problems=["addition", "multiplication", "division", "modulo", "gcd", "lcm", "square_mod", "bitwise_and", "bitwise_xor", "bitwise_or"]

SE = SymbolicEngine(VSA_dim=VSA_dim, max_digits=max_digits, possible_problems=possible_problems, 
                    curr_dir=curr_dir)
possible_problems_str = "_".join(possible_problems)
torch.save(SE, f"{curr_dir}/VSA_library/symbolic_engine_VSA_dim_{VSA_dim}_max_digits_{max_digits}_problem_types_{possible_problems_str}.pt")

if type(problem_type) == type([]):
    problem_str = "_".join(problem_type)
else:
    problem_str = problem_type

# Output directory to write and read saved hidden state and VSA data to
save_dir = f"{curr_dir}/gathered_data_{complexity}_complexity_{tokens_to_keep}_tokens_kept_{calculate_end_index}_calculating_end_index_{train_data_rounds}_train_rounds_{test_data_rounds}_test_rounds_{problem_str}"
os.makedirs(save_dir, exist_ok=True)

if generate_data:
    generate_and_save_data(generator=generator, SE=SE, save_dir=save_dir, 
                           rounds=train_data_rounds, train=True,  save_frequency=save_frequency,
                           complexity=complexity, n_samples=n_samples, problem_type=problem_type,
                           tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True)
    print("Training data gathering completed and saved to disk.")

    generate_and_save_data(generator=generator, SE=SE, save_dir=save_dir, 
                           rounds=test_data_rounds,  train=False, save_frequency=save_frequency,
                           complexity=complexity, n_samples=n_samples, problem_type=problem_type,
                           tokens_to_keep=tokens_to_keep, calculate_end_index=calculate_end_index, verbose=True)
    print("Testing data gathering completed and saved to disk.")

if not generate_data:
    training_encoder_data_loaders = generate_data_loaders(train=True, save_dir=save_dir, data_rounds=train_data_rounds, 
                                                          save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=restrict_train_dataset, 
                                                          tokens_to_keep=tokens_to_keep, batch_size=encoder_decoder_training_batch_size, verbose=True)
    print("Training data loaders for each layer have been created.")
    testing_encoder_data_loaders = generate_data_loaders(train=False, save_dir=save_dir, data_rounds=test_data_rounds, 
                                                         save_frequency=save_frequency, layer_numbers=layer_numbers, restrict_dataset=restrict_test_dataset, 
                                                         tokens_to_keep=tokens_to_keep, batch_size=encoder_decoder_training_batch_size, verbose=True)
    print("Testing data loaders for each layer have been created.")

    encoders = torch.nn.ModuleList()
    for layer_id in layer_numbers:
        if tokens_to_keep == 1:
            #layer_encoder = Encoder_Deep(layer_id, model_dim, VSA_dim, model_dim*4).to(device)
            layer_encoder = Encoder(layer_id, model_dim, VSA_dim, ).to(device)
        else:
            layer_encoder = LastTokenTransformer(layer_id, model_dim, VSA_dim, num_layers=2, hidden_dim=512).to(device)
        encoders.append(layer_encoder)#, dtype=torch.float32))

    print("Trainable parameters per layer:", count_trainable_parameters(layer_encoder)) # Per Layer

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
            for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
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

    plt.plot(losses.mean(axis=0)[:i], marker=".")
    plt.title("Average RMSE Loss Per Epoch")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    if log_wandb:
        wandb.log({f"Average RMSE Loss Per Epoch": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()
            

    plt.plot(losses.T[:i].mean(axis=0), marker=".")
    x_ticks = np.arange(0, len(losses.T.mean(axis=0)[:-1]), 1)
    plt.xticks(x_ticks, rotation=75)
    plt.title("Average RMSE Loss vs Layer Number")
    plt.ylabel("Final RMSE Loss")
    plt.xlabel("Layer Number")
    if log_wandb:
        wandb.log({f"Final RMSE Loss vs Layer Number": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    plt.plot(losses.T[i], marker=".")
    x_ticks = np.arange(0, len(losses.T.mean(axis=0)[:-1]), 1)
    plt.xticks(x_ticks, rotation=75)
    plt.title("Final RMSE Loss vs Layer Number")
    plt.ylabel("Final RMSE Loss")
    plt.xlabel("Layer Number")
    if log_wandb:
        wandb.log({f"Final RMSE Loss vs Layer Number": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    ####################################################################################################

    for n, n_layer in enumerate(layer_numbers):
        for param in encoders[n].parameters():
            param.requires_grad = False



    decoders = torch.nn.ModuleList()
    for layer_id in layer_numbers:
        #layer_decoder = Decoder_Deep(layer_id, VSA_dim, model_dim, model_dim*2).to(device)
        layer_decoder = Decoder(layer_id, VSA_dim, model_dim).to(device)
        decoders.append(layer_decoder)#, dtype=torch.float32))


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
            for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
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
    if log_wandb:
        wandb.log({f"Average Decoder RMSE Loss Per Epoch": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    plt.plot(decoding_losses.T[:j].mean(axis=0), marker=".")
    x_ticks = np.arange(0, len(decoding_losses.T.mean(axis=0)[:-1]), 1)
    plt.xticks(x_ticks, rotation=75)
    plt.title("Average Decoder RMSE Loss vs Layer Number")
    plt.ylabel("Average RMSE Loss")
    plt.xlabel("Layer Number")
    if log_wandb:
        wandb.log({f"Average Decoder RMSE Loss vs Layer Number": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    plt.plot(decoding_losses.T[j], marker=".")
    x_ticks = np.arange(0, len(decoding_losses.T.mean(axis=0)[:-1]), 1)
    plt.xticks(x_ticks, rotation=75)
    plt.title("Final Decoder RMSE Loss vs Layer Number")
    plt.ylabel("Final RMSE Loss")
    plt.xlabel("Layer Number")
    if log_wandb:
        wandb.log({f"Final Decoder RMSE Loss vs Layer Number": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()


    if not os.path.exists(f"{curr_dir}/models"):
        os.mkdir(f"{curr_dir}/models")
    torch.save(encoders.state_dict(), f"{curr_dir}/models/encoders_state_dict_{curr_date}.pth")
    torch.save(encoders,              f"{curr_dir}/models/encoders_{curr_date}.pth")
    print("Saved:", f"{curr_dir}/models/encoders_state_dict_{curr_date}.pth", "and", f"models/encoders_{curr_date}.pth")
    torch.save(decoders.state_dict(), f"{curr_dir}/models/decoders_state_dict_{curr_date}.pth")
    torch.save(decoders,              f"{curr_dir}/models/decoders_{curr_date}.pth")
    print("Saved:", f"{curr_dir}/models/decoders_state_dict_{curr_date}.pth", "and", f"models/decoders_{curr_date}.pth")

    ####################################################################################################
    
    print("Running Error Statistics for Testing Data")

    #errors_per_pt = {layer_num.item() : {pt: [] for pt in possible_problems} for n, layer_num in enumerate(layer_numbers)}
    errors_per_pt = {pt: {layer_num.item(): [] for n, layer_num in enumerate(layer_numbers)} for pt in problem_type}

    VSA_predictions = []
    label_VSAs      = []
    rows_to_print  = 0
    verbose = 1 # 0, 1, 2
    errors = np.zeros(len(layer_numbers))
    lowest_error_layer = 0
    lowest_error       = complexity + 1
    lowest_pt_error    = np.inf
    calculate_digit_error = True # Measure total number of errors per digit
    per_digit_errors = torch.zeros((len(layer_numbers), SE.max_digits))
    exponents = torch.tensor([10 ** i for i in range(SE.max_digits)])
    with torch.no_grad():
        for n, layer in enumerate(layer_numbers):
            row_count = 0
            if verbose:
                print("--------- Layer", layer.item(), "---------")
            e = 0
            for batch_idx, (data, labels) in enumerate(testing_encoder_data_loaders[n]):
                row_count += len(data)
                pred = encoders[n](data)
                decoded_n1 = (SE.decode_digits(pred.  type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
                decoded_n2 = (SE.decode_digits(pred.  type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)
                actual_n1  = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
                actual_n2  = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)
                decoded_problem_types, _, _ = SE.decode_problem_type(pred)
                actual_problem_types, _, _  = SE.decode_problem_type(labels)
                if calculate_digit_error:
                    n1_batch_error, n1_per_digit_error = SE.digit_error(decoded_n1, actual_n1, error_per_digit=calculate_digit_error, verbose=verbose)
                    n2_batch_error, n2_per_digit_error = SE.digit_error(decoded_n2, actual_n2, error_per_digit=calculate_digit_error, verbose=verbose)
                    batch_error = (n1_batch_error + n2_batch_error) / 2 * len(data) / (test_data_rounds * max_batch_size)
                    per_digit_error = (n1_per_digit_error + n2_per_digit_error) / 2 * len(data) / (test_data_rounds * max_batch_size)
                    #print(per_digit_error)
                    per_digit_errors[layer.item()] += per_digit_error
                else:
                    batch_error = SE.digit_error(decoded_n1, actual_n1, verbose=verbose) + SE.digit_error(decoded_n2, actual_n2, verbose=verbose)
                    batch_error = batch_error / 2 * len(data) / (test_data_rounds * max_batch_size)
                for k, curr_pt in enumerate(actual_problem_types):
                    errors_per_pt[curr_pt][layer.item()] += [batch_error[k].item()]
                for r in range(rows_to_print):
                    print("Decoded symbolic encodings: first number:",  decoded_n1[r], "second number:", decoded_n2[r])
                    print("Actual           encodings: first number:",  actual_n1[r],  "second number:", actual_n2[r])
                    print("Decoded problem type:", decoded_problem_types[r])
                    print("Actual  problem type:", actual_problem_types[r])
                    #print(" --------- Error:", decoded_n1[r]-actual_n1[r], decoded_n2[r]-actual_n2[r], )
                e += batch_error.float().mean().item() 

            errors[n] = e
            per_digit_errors[layer.item()] = per_digit_errors[layer.item()] / row_count
            problem_type_error = (decoded_problem_types != actual_problem_types).sum()
            if e < lowest_error:
                lowest_error_layer = layer
                lowest_error       = e
                lowest_pt_error    = problem_type_error

            print("Average Error:", errors[n], "digits out of", complexity+1)
            print("Average Problem Type Error:", problem_type_error, "out of", len(labels))
            # Divide per_digit_errors by row_count and by 2 in order to get per digit error
            print("Average Error Rate per Digit:", per_digit_errors[layer.item()])


    x_ticks = np.arange(layer_numbers.cpu()[0].item(), layer_numbers.cpu()[0].item() + len(layer_numbers), 1)
    plt.plot(x_ticks, errors, marker=".")
    plt.xticks(x_ticks, rotation=75)
    plt.title("Error of Decoded Numbers (Testing Data)")
    plt.ylabel("Average Number of Incorrectly Decoded Digits")
    plt.xlabel("Layer Number")
    plt.grid(False)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers (Testing Data)": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    for pt in problem_type:
        x_ticks = np.arange(layer_numbers.cpu()[0].item(), layer_numbers.cpu()[0].item() + len(layer_numbers), 1)
        pt_error = [np.mean(errors_per_pt[pt][ln]) for ln in errors_per_pt[pt]]
        plt.plot(x_ticks, pt_error, marker=".")
        plt.xticks(x_ticks, rotation=75)
        #plt.close()
    plt.title(f"Error of Decoded Numbers per Problem Type (Testing Data)")
    plt.ylabel("Mean Absolute Decoding Error (Testing Data)")
    plt.xlabel("Layer Number")
    plt.legend(problem_type)
    plt.grid(False)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers per Problem Type (Testing Data)": wandb.Image(plt)})
    else:
        plt.show()
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

    for n, digit in enumerate(per_digit_errors.float().cpu().numpy().T):
        plt.plot(layer_numbers.cpu(), digit, label=labels[n], marker=markers[n])

    plt.title(f"Error of Decoded Numbers per Digit (Testing Data)")
    plt.xlabel('Layer Number')
    plt.ylabel('Classification Error Rate')
    plt.legend()
    plt.grid(True)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers per Digit (Testing Data)": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    print("Minimum Error:", lowest_error, "and problem type error:", problem_type_error, "at layer", lowest_error_layer.item())#, ", Current running loss:", running_losses[lowest_error_layer.item()])

    ####################################################################################################
    
    print("Running Error Statistics for Training Data")

    errors_per_pt = {pt: {layer_num.item(): [] for n, layer_num in enumerate(layer_numbers)} for pt in problem_type}

    VSA_predictions = []
    label_VSAs      = []
    rows_to_print  = 0
    verbose = 1 # 0, 1, 2
    errors = np.zeros(len(layer_numbers))
    lowest_error_layer = 0
    lowest_error       = complexity + 1
    lowest_pt_error    = np.inf
    calculate_digit_error = True # Measure total number of errors per digit
    per_digit_errors = torch.zeros((len(layer_numbers), SE.max_digits))
    exponents = torch.tensor([10 ** i for i in range(SE.max_digits)])
    with torch.no_grad():
        for n, layer in enumerate(layer_numbers):
            row_count = 0
            if verbose:
                print("--------- Layer", layer.item(), "---------")
            e = 0
            for batch_idx, (data, labels) in enumerate(training_encoder_data_loaders[n]):
                row_count += len(data)
                pred = encoders[n](data)
                decoded_n1 = (SE.decode_digits(pred.  type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
                decoded_n2 = (SE.decode_digits(pred.  type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)
                actual_n1  = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n1]), SE.VSA_n1) * exponents).sum(axis=1)
                actual_n2  = (SE.decode_digits(labels.type_as(SE.vectors[SE.VSA_n2]), SE.VSA_n2) * exponents).sum(axis=1)
                decoded_problem_types, _, _ = SE.decode_problem_type(pred)
                actual_problem_types, _, _  = SE.decode_problem_type(labels)
                if calculate_digit_error:
                    n1_batch_error, n1_per_digit_error = SE.digit_error(decoded_n1, actual_n1, error_per_digit=calculate_digit_error, verbose=verbose)
                    n2_batch_error, n2_per_digit_error = SE.digit_error(decoded_n2, actual_n2, error_per_digit=calculate_digit_error, verbose=verbose)
                    batch_error = (n1_batch_error + n2_batch_error) / 2 * len(data) / (train_data_rounds * max_batch_size)
                    per_digit_error = (n1_per_digit_error + n2_per_digit_error) / 2 * len(data) / (train_data_rounds * max_batch_size)
                    #print(per_digit_error)
                    per_digit_errors[layer.item()] += per_digit_error
                else:
                    batch_error = SE.digit_error(decoded_n1, actual_n1, verbose=verbose) + SE.digit_error(decoded_n2, actual_n2, verbose=verbose)
                    batch_error = batch_error / 2 * len(data) / (train_data_rounds * max_batch_size)
                for k, curr_pt in enumerate(actual_problem_types):
                    errors_per_pt[curr_pt][layer.item()] += [batch_error[k].item()]
                for r in range(rows_to_print):
                    print("Decoded symbolic encodings: first number:",  decoded_n1[r], "second number:", decoded_n2[r])
                    print("Actual           encodings: first number:",  actual_n1[r],  "second number:", actual_n2[r])
                    print("Decoded problem type:", decoded_problem_types[r])
                    print("Actual  problem type:", actual_problem_types[r])
                    #print(" --------- Error:", decoded_n1[r]-actual_n1[r], decoded_n2[r]-actual_n2[r], )
                e += batch_error.float().mean().item() 

            errors[n] = e
            per_digit_errors[layer.item()] = per_digit_errors[layer.item()] / row_count
            problem_type_error = (decoded_problem_types != actual_problem_types).sum()
            if e < lowest_error:
                lowest_error_layer = layer
                lowest_error       = e
                lowest_pt_error    = problem_type_error

            print("Average Error:", errors[n], "digits out of", complexity+1)
            print("Average Problem Type Error:", problem_type_error, "out of", len(labels))
            # Divide per_digit_errors by row_count and by 2 in order to get per digit error
            print("Average Error Rate per Digit:", per_digit_errors[layer.item()])


    x_ticks = np.arange(layer_numbers.cpu()[0].item(), layer_numbers.cpu()[0].item() + len(layer_numbers), 1)
    plt.plot(x_ticks, errors, marker=".")
    plt.xticks(x_ticks, rotation=75)
    plt.title("Error of Decoded Numbers (Training Data)")
    plt.ylabel("Average Number of Incorrectly Decoded Digits")
    plt.xlabel("Layer Number")
    plt.grid(False)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers (Training Data)": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()

    for pt in problem_type:
        x_ticks = np.arange(layer_numbers.cpu()[0].item(), layer_numbers.cpu()[0].item() + len(layer_numbers), 1)
        pt_error = [np.mean(errors_per_pt[pt][ln]) for ln in errors_per_pt[pt]]
        plt.plot(x_ticks, pt_error, marker=".")
        plt.xticks(x_ticks, rotation=75)
        #plt.close()
    plt.title(f"Error of Decoded Numbers per Problem Type (Training Data)")
    plt.ylabel("Mean Absolute Decoding Error")
    plt.xlabel("Layer Number")
    plt.legend(problem_type)
    plt.grid(False)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers per Problem Type (Training Data)": wandb.Image(plt)})
    else:
        plt.show()
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

    for n, digit in enumerate(per_digit_errors.float().cpu().numpy().T):
        plt.plot(layer_numbers.cpu(), digit, label=labels[n], marker=markers[n])

    plt.title(f"Error of Decoded Numbers per Digit (Training Data)")
    plt.xlabel('Layer Number')
    plt.ylabel('Classification Error Rate')
    plt.legend()
    plt.grid(True)
    if log_wandb:
        wandb.log({f"Error of Decoded Numbers per Digit (Training Data)": wandb.Image(plt)})
    else:
        plt.show()
    plt.close()
    
    if log_wandb:
        wandb.finish()


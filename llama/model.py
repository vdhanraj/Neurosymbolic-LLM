# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from llama.vsa_engine import SymbolicEngine, bind

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_scaled_rope: bool = True

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, curr_token=0, curr_pt="addition", curr_x=0, curr_y=0, verbose=False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        h_stack = []
        for n, layer in enumerate(self.layers):
            h_stack += [h.clone()]
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        h_stack += [h.clone()]
        h_stack = torch.stack(h_stack)
        #print(h_stack.shape, h.shape)
        output = self.output(h).float()
        return output, h_stack, h

    #@torch.inference_mode()
    def forward_symbolic_funnel(self, tokens: torch.Tensor, start_pos: int, curr_token=0, curr_pt="addition", curr_x=0, curr_y=0, verbose=False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Shape of h is batch_size, token_number, hidden_dim
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        skip_index = 0 # Indicates which skip connection weight is being used (different ones for different layers if it's trainable)
        for n, layer in enumerate(self.layers):
            if n == self.symbolic_encoding_layer and (self.multi_token_intervention or curr_token == 0) and not self.bypass_symbolic and not self.add_noise:
                if curr_token == 0:
                    appended_h = h
                    if self.encoder_input_tokens == 1:
                        relevant_h = appended_h[:,-1,:] # relevant_h will be of shape (batch_size, h_dim)
                    elif self.encoder_input_tokens == "all":
                        start_indices, end_indices = self.curr_start_indices, self.curr_end_indices # use the start and end indices for the current problems
                        if self.calculate_end_index:
                            relevant_h = [appended_h[b, start_indices[b]:end_indices[b],:] for b in range(appended_h.shape[0])] # find the tokens corresponding to the start and end indices per batch example
                            max_tok_length = max([h.shape[0] for h in relevant_h]) # find the maximum length input for each reduced batch example
                            relevant_h = torch.stack([F.pad(h, (max_tok_length - h.shape[1], 0, 0, 0)) for h in relevant_h]) # pad each reduced batch example with zeros so that they are all the same size, then stack them. Final shape is (batch_size, max_tok_length, h_dim)
                        else: # Same as previous block, but in this case we keep tokens from the start index to the final token
                            relevant_h = [appended_h[b, start_indices[b]:,:] for b in range(appended_h.shape[0])]
                            max_tok_length = max([h.shape[0] for h in relevant_h])
                            relevant_h = torch.stack([F.pad(h, (0, 0, max_tok_length - h.shape[0], 0)) for h in relevant_h])
                    else:
                        relevant_h = h[:, -self.encoder_input_tokens:, :] # relevant_h will be of shape (batch_size, encoder_input_tokens, h_dim)
                else:
                    if self.encoder_input_tokens == 1:
                        relevant_h = h[:,-1,:]
                    else:
                        appended_h = torch.cat([self.relevant_h, h], dim=1) # If curr_token > 0, then reuse the previously caclulated relevent h, and append the current h to it (along the token dimension)
                        relevant_h = appended_h
                self.relevant_h = relevant_h # Save to recalculate relevant_h for next token

                symbolic_encoding = self.encoders[n-self.starting_encoder_layer](relevant_h) # Pass the relevent_h through the symbolic encoder

                # If using lora baseline, simply set the output of the encoder to the input of the decoder without doing any symbolic computation
                if self.lora_baseline:
                    final_symbol = symbolic_encoding
                    use_symbolic_layer = torch.tensor([True for _ in range(_bsz)], device=h.device).view(-1, 1)
                    print("Using LoRA")
                else:
                    if not self.static_encoding or curr_token == 0: # Execute if the curr_token is 0, or if static_encoding is set to False (in which case you recalculate final_symbol during each forward pass)
                        problem_type_identity = {"addition": lambda x: 0, "multiplication": lambda x: 1,   "modulo":      lambda x: x+1, "gcd":         lambda x: x, 
                                                 "lcm":      lambda x: x, "square_mod":     lambda x: x+1, "bitwise_and": lambda x: x,   "bitwise_xor": lambda x: 0, "bitwise_or": lambda x: 0}

                        #print(symbolic_encoding.shape)
                        # TODO: Change the problem_subset to be all problems when doing testing (i.e., when the actual problem type would not be included in the training_problems) # Done
                        problem_type_decoded, problem_type_score, score_per_problem = self.SE.decode_problem_type(symbolic_encoding, problem_subset=self.SE.possible_problems, normalize_VSA_before_dot=self.normalize_VSA_before_dot)
                        score_per_problem = [{self.SE.possible_problems[i]: spp[i] for i in range(len(spp))} for spp in score_per_problem]
                        use_symbolic_layer = torch.tensor([bool(i > self.problem_score_threshold) for i in problem_type_score], device=h.device).view(-1, 1) 
                        if verbose == 2:
                            print("Decoded problem type, max score, score above threshold:", problem_type_decoded, problem_type_score, use_symbolic_layer)
                            print("Score per problem type:", score_per_problem)
                        problem_type = problem_type_decoded[0] # Assume all items in the batch have the same decoded problem type in order to do effecient batch processing
                        # TODO: Get this to work for when the batch has different problem types present. Current workaround is to make n_samples = 1 if you can't gaurentee that the batch has only 1 type of problem type

                        if self.simulate_perfect_encoder:
                            problem_type = curr_pt

                        if self.record_score_per_problem == 1 and self.current_split == "train":
                            with open(f"{self.curr_dir}/outputs/score_per_problem_training_{self.wandb_run_id}.txt", "a") as file:
                                for batch in score_per_problem:
                                    for k in batch:
                                        file.write(self.current_split + "," + curr_pt + "," + k + "," + str(batch[k]) + "\n")
                        if self.record_score_per_problem == 2 and self.current_split == "test":
                            with open(f"{self.curr_dir}/outputs/score_per_problem_testing_{self.wandb_run_id}.txt", "a") as file:
                                for batch in score_per_problem:
                                    for k in batch:
                                        file.write(self.current_split + "," + curr_pt + "," + k + "," + str(batch[k]) + "\n")
                        if self.record_score_per_problem == 3 and (self.current_split == "train" or self.current_split == "test"):
                            with open(f"{self.curr_dir}/outputs/score_per_problem_training_and_testing_{self.wandb_run_id}.txt", "a") as file:
                                for batch in score_per_problem:
                                    for k in batch:
                                        file.write(self.current_split + "," + curr_pt + "," + k + "," + str(batch[k]) + "\n")

                        if not self.simulate_perfect_encoder:
                            decoded_n1 = (self.SE.decode_digits(symbolic_encoding.type_as(self.SE.vectors[self.SE.VSA_n1]), self.SE.VSA_n1) * torch.tensor([10 ** i for i in range(self.SE.max_digits)])).sum(axis=1).tolist()
                            decoded_n2 = (self.SE.decode_digits(symbolic_encoding.type_as(self.SE.vectors[self.SE.VSA_n2]), self.SE.VSA_n2) * torch.tensor([10 ** i for i in range(self.SE.max_digits)])).sum(axis=1).tolist()
                        else:
                            decoded_n1 = curr_x
                            decoded_n2 = curr_y
                        if problem_type == "addition":
                            # For addition, we can use the purely differentiable method we created (we don't have to though)
                            # symbolic_sums = [self.SE.add_VSA(symbolic_encoding[i].unsqueeze(0).type_as(self.SE.vectors[self.SE.VSA_n1])) for i in range(_bsz)]
                            # symbolic_sums = [bind(symbolic_sums[i], self.SE.vectors[self.SE.VSA_n1]).type_as(h) for i in range(_bsz)]
                            # final_symbol = torch.stack(symbolic_sums)

                            decoded_sums  = torch.tensor([int(decoded_n1[i] + decoded_n2[i]) for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_sums, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_sums, [problem_type_identity[problem_type](x) for x in decoded_sums], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)

                            if verbose:
                                print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "sum:", decoded_n1[0]+decoded_n2[0])
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "sum:", decoded_n1[k]+decoded_n2[k])
                        elif problem_type == "multiplication":
                            if self.limit_solution_digits:
                                decoded_prods  = torch.tensor([int(decoded_n1[i] * decoded_n2[i]) % 10**(self.complexity+1) for i in range(_bsz)])
                            else:
                                decoded_prods  = torch.tensor([int(decoded_n1[i] * decoded_n2[i])                           for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_prods, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_prods, [problem_type_identity[problem_type](x) for x in decoded_prods], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            if verbose:
                                if self.limit_solution_digits:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "product:", decoded_n1[0]*decoded_n2[0] % 10**(self.complexity+1))
                                else:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "product:", decoded_n1[0]*decoded_n2[0])
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    if self.limit_solution_digits:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "product:", decoded_n1[k]*decoded_n2[k] % 10**(self.complexity+1))
                                    else:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "product:", decoded_n1[k]*decoded_n2[k])

                        elif problem_type == "division":
                            # decoded_quots = []
                            # for i in range(_bsz):
                            #     if decoded_n2[i] != 0:
                            #         decoded_quot = int(decoded_n1[i] // decoded_n2[i])
                            #     else:
                            #         decoded_quot = 0
                            #     decoded_quots += [decoded_quot]
                            # decoded_quots = torch.tensor(decoded_quots)
                            decoded_quots = torch.tensor([int(decoded_n1[i] // decoded_n2[i]) if decoded_n2[i] != 0 else 0 for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_quots, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_quots, [problem_type_identity[problem_type](x) for x in decoded_quots], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            if verbose:
                                if decoded_n2[0] != 0:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "quotient:", decoded_n1[0]/decoded_n2[0])
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    if decoded_n2[k] != 0:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "quotient:", decoded_n1[k]/decoded_n2[k])

                        elif problem_type == "modulo":
                            # decoded_mods = []
                            # for i in range(_bsz):
                            #     if decoded_n2[i] != 0:
                            #         decoded_mod = int(decoded_n1[i] % decoded_n2[i])
                            #     else:
                            #         decoded_mod = 0
                            #     decoded_mods += [decoded_mod]
                            # decoded_mods = torch.tensor(decoded_mod)
                            decoded_mods = torch.tensor([
                                int(decoded_n1[i] % decoded_n2[i]) if decoded_n2[i] != 0 else 0
                                for i in range(_bsz)
                            ])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_mods, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_mods, [problem_type_identity[problem_type](x) for x in decoded_mods], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)

                            if verbose:
                                if decoded_n2[0] != 0:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "modulo:", decoded_n1[0]%decoded_n2[0])
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    if decoded_n2[k] != 0:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "modulo:", decoded_n1[k]%decoded_n2[k])

                        elif problem_type == "gcd":
                            # decoded_gcds = []
                            # for i in range(_bsz):
                            #     decoded_gcd = np.gcd(int(decoded_n1[i]), int(decoded_n2[i]))
                            #     decoded_gcds += [int(decoded_gcd)]
                            # decoded_gcds = torch.tensor(decoded_gcds)
                            decoded_gcds = torch.tensor([
                                int(np.gcd(int(decoded_n1[i]), int(decoded_n2[i])))
                                for i in range(_bsz)
                            ])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_gcds, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_gcds, [problem_type_identity[problem_type](x) for x in decoded_gcds], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            if verbose:
                                print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "gcd:", np.gcd(int(decoded_n1[0]), int(decoded_n2[0])))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "gcd:", np.gcd(int(decoded_n1[k]), int(decoded_n2[k])))

                        elif problem_type == "lcm":
                            # decoded_lcms = []
                            # for i in range(_bsz):
                            #     if self.limit_solution_digits:
                            #         decoded_lcm  = np.lcm(int(decoded_n1[i]), int(decoded_n2[i])) % 10**(self.complexity+1)
                            #     else:
                            #         decoded_lcm  = np.lcm(int(decoded_n1[i]), int(decoded_n2[i]))
                            #     decoded_lcms += [int(decoded_lcm)]
                            # decoded_lcms = torch.tensor(decoded_lcms)
                            decoded_lcms = torch.tensor([
                                int(np.lcm(int(decoded_n1[i]), int(decoded_n2[i])) % 10**(self.complexity+1)) if self.limit_solution_digits
                                else int(np.lcm(int(decoded_n1[i]), int(decoded_n2[i])))
                                for i in range(_bsz)
                            ])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_lcms, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_lcms, [problem_type_identity[problem_type](x) for x in decoded_lcms], single_number_generation=False, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            if verbose:
                                if self.limit_solution_digits:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "lcm:", np.lcm(int(decoded_n1[0]), int(decoded_n2[0])) % 10**(self.complexity+1))
                                else:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "lcm:", np.lcm(int(decoded_n1[0]), int(decoded_n2[0])))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    if self.limit_solution_digits:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "lcm:", np.lcm(int(decoded_n1[k]), int(decoded_n2[k])) % 10**(self.complexity+1))
                                    else:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "lcm:", np.lcm(int(decoded_n1[k]), int(decoded_n2[k])))

                        elif problem_type == "square_mod":
                            # decoded_sqs  = []
                            # for i in range(_bsz):
                            #     if decoded_n2[i] != 0:
                            #         decoded_sq = int(decoded_n1[i])**2 % int(decoded_n2[i])
                            #     else:
                            #         decoded_sq = 0
                            #     decoded_sqs += [decoded_sq]
                            # decoded_sqs = torch.tensor(decoded_sqs)
                            decoded_sqs = torch.tensor([
                                int(decoded_n1[i])**2 % int(decoded_n2[i]) if decoded_n2[i] != 0 else 0
                                for i in range(_bsz)
                            ])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_sqs, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_sqs, [problem_type_identity[problem_type](x) for x in decoded_sqs], single_number_generation=False, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            if verbose:
                                if decoded_n2[0] != 0:
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "square_mod:", int(decoded_n1[0])**2 % int(decoded_n2[0]))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    if decoded_n2[k] != 0:
                                        print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "square_mod:", int(decoded_n1[k])**2 % int(decoded_n2[k]))

                        elif problem_type == "bitwise_and":
                            decoded_and  = torch.tensor([int(decoded_n1[i]) & int(decoded_n2[i]) for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_and, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_and, [problem_type_identity[problem_type](x) for x in decoded_and], single_number_generation=False, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            if verbose:
                                print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "bitwise_and:", int(decoded_n1[0]) & int(decoded_n2[0]))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "bitwise_and:", int(decoded_n1[k]) & int(decoded_n2[k]))

                        elif problem_type == "bitwise_xor":
                            decoded_xor  = torch.tensor([int(decoded_n1[i]) ^ int(decoded_n2[i]) for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_xor, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_xor, [problem_type_identity[problem_type](x) for x in decoded_xor], single_number_generation=False, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            if verbose:
                                print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "bitwise_xor:", int(decoded_n1[0]) ^ int(decoded_n2[0]))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "bitwise_xor:", int(decoded_n1[k]) ^ int(decoded_n2[k]))

                        elif problem_type == "bitwise_or":
                            decoded_or  = torch.tensor([int(decoded_n1[i]) | int(decoded_n2[i]) for i in range(_bsz)])
                            if not self.use_specific_identities:
                                final_symbol = self.SE.generate_VSA(decoded_or, torch.zeros(_bsz).to(torch.int), single_number_generation=self.single_number_generation, problem_types=[problem_type]*_bsz).to(torch.bfloat16)
                            else:
                                final_symbol = self.SE.generate_VSA(decoded_or, [problem_type_identity[problem_type](x) for x in decoded_or], single_number_generation=False, problem_types=[problem_type] * _bsz).to(torch.bfloat16)
                            if verbose:
                                print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[0]), "second number:", int(decoded_n2[0]), "bitwise_or:", int(decoded_n1[0]) | int(decoded_n2[0]))
                            if verbose == 2:
                                for k in range(1, _bsz):
                                    print(f"Decoded symbolic encodings (pass # {curr_token+1}): first number:",  int(decoded_n1[k]), "second number:", int(decoded_n2[k]), "bitwise_or:", int(decoded_n1[k]) | int(decoded_n2[k]))

                        if self.static_encoding: # If static_encoding is set to True, then we need to save final_symbol and use_symbolic_layer for future use
                            self.final_symbol = final_symbol
                            self.use_symbolic_layer = use_symbolic_layer

                        if self.calculate_encoding_accuracy:
                            for k in range(_bsz):
                                for digit in range(self.complexity + 1):
                                    self.encoding_accuracy[curr_pt]["digit " + str(digit)]["first_number"]  += [str(int(decoded_n1[k])).zfill(self.complexity+1)[::-1][digit] == str(curr_x[k]).zfill(self.complexity+1)[::-1][digit]]
                                    self.encoding_accuracy[curr_pt]["digit " + str(digit)]["second_number"] += [str(int(decoded_n2[k])).zfill(self.complexity+1)[::-1][digit] == str(curr_y[k]).zfill(self.complexity+1)[::-1][digit]]
                    
                    elif self.static_encoding: # If the curr_token is not 0 and we are using static_encoding, then just load the final_symbol and use_symbolic_layer as computed before
                        final_symbol       = self.final_symbol.clone()
                        use_symbolic_layer = self.use_symbolic_layer
                    #print(final_symbol.shape, self.SE.generate_counter(curr_token, _bsz).shape, final_symbol.dtype, self.SE.generate_counter(curr_token, _bsz).dtype)
                    if self.encode_counter: 
                        final_symbol = final_symbol + self.SE.generate_counter(curr_token, _bsz) # Encode information corresponding to which token is currently being decoded

            if n in self.symbolic_decoding_layers and (self.multi_token_intervention or curr_token == 0) and not self.bypass_symbolic:
                if not self.add_noise:
                    modified_h = self.decoders[n-self.starting_decoder_layer](final_symbol)
                    modified_h = modified_h.reshape(_bsz, self.params.dim)
                if self.add_noise:
                    modified_h = torch.randn_like(h[:,-1,:])
                if self.normalize_vector:
                    # Make sure the modified_h vector has the same statistical properties as the h we are adding it onto
                    modified_h = (modified_h - modified_h.mean()) / modified_h.std()
                    modified_h = modified_h * h[:,-1,:].mean() + h[:,-1,:].std()
                if not self.rms_layer:
                    # Concatenate the modified token of the LLM (the most recent token) with the rest of the tokens
                    h = torch.cat([h[:, :-1, :], torch.where(use_symbolic_layer, modified_h * (1 - self.skip_weights[skip_index]) + h[:, -1, :] * self.skip_weights[skip_index], h[:, -1, :]).unsqueeze(1)], dim=1)
                else:
                    #h = torch.cat([h[:, :-1, :], (modified_h + h[:, -1, :]).unsqueeze(1)], dim=1)
                    h = torch.cat([h[:, :-1, :], torch.where(use_symbolic_layer, modified_h + h[:, -1, :], h[:, -1, :]).unsqueeze(1)], dim=1)
                    
                    h = self.rms_layers[skip_index](h)
                skip_index = skip_index + 1

            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        n = n + 1
    
        # Intervene after the final layer
        if n in self.symbolic_decoding_layers and (self.multi_token_intervention or curr_token == 0) and not self.bypass_symbolic:
            if not self.add_noise:
                modified_h = self.decoders[n-self.starting_decoder_layer](final_symbol)
                modified_h = modified_h.reshape(_bsz, self.params.dim)
            if self.add_noise:
                modified_h = torch.randn_like(h[:,-1,:])
            if self.normalize_vector:
                # Make sure the modified_h vector has the same statistical properties as the h we are adding it onto
                modified_h = (modified_h - modified_h.mean()) / modified_h.std()
                modified_h = modified_h * h[:,-1,:].mean() + h[:,-1,:].std()
            if not self.rms_layer:
                # Concatenate the modified token of the LLM (the most recent token) with the rest of the tokens
                h = torch.cat([h[:, :-1, :], torch.where(use_symbolic_layer, modified_h * (1 - self.skip_weights[skip_index]) + h[:, -1, :] * self.skip_weights[skip_index], h[:, -1, :]).unsqueeze(1)], dim=1)
            else:
                h = torch.cat([h[:, :-1, :], (modified_h + h[:, -1, :]).unsqueeze(1)], dim=1)
                h = self.rms_layers[skip_index](h)
            skip_index = skip_index + 1

        n = n + 1
        output = self.output(h).type_as(h)

        # Intervene at the output projection layer 
        if (self.multi_token_intervention or curr_token == 0) and not self.bypass_symbolic and n in self.symbolic_decoding_layers:
            modified_output = self.decoders[n-self.starting_decoder_layer](final_symbol)
            if not self.rms_layer:
                output = torch.cat([output[:, :-1, :], torch.where(use_symbolic_layer, modified_output * (1 - self.skip_weights[skip_index]) + output[:, -1, :] * self.skip_weights[skip_index], output[:, -1, :]).unsqueeze(1)], dim=1)
            else:
                output = torch.cat([output[:, :-1, :], (modified_output + output[:, -1, :]).unsqueeze(1)], dim=1)
                output = self.rms_layers[skip_index](output)
            skip_index = skip_index + 1

        return output, None, None

    @torch.inference_mode()
    def forward_no_tok_embedding(self, h: torch.Tensor, start_pos: int):
        seqlen = h.shape[0]
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None

        for n, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()
        return output, h

    @torch.inference_mode()
    def forward_forced(self, tokens: torch.Tensor, start_pos: int, h_hat, gain=0.1):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        error = h - h_hat
        h = h - error * gain
        output = self.output(h).float()
        return output, h

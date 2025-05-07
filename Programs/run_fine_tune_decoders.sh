#!/bin/bash
    
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=120:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=CELIASMI

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o slurm_outputs/JOB%j.out # File to which STDOUT will be written
#SBATCH -e slurm_errors/JOB%j-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=vdhanraj@uwaterloo.ca
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate torch
 
# Task to run
# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_2 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.2 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_3 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.3 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_4 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.4 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_5 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_6 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.6 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_7 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.7 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_8 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.8 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29500 --run_name all_pts_run2_3_digits_baseline_thresh_9 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.9 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth




# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_5 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.5 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_6 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.6 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_7 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.7 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_8 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.8 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_9 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.9 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_4 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.4 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_3 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.3 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_2 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.2 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name all_pts_run2_3_digits_baseline_skip_weight_1 --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
#                                                           --problem_score_threshold 0.5 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
#                                                           --trainable_skip 1 --starting_skip_strength 0.1 --multi_token_intervention 1 --encode_counter 1 \
#                                                           --verbose 2\
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth



# # Task to run
python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29502 --run_name all_pts_run2_3_digits_baseline_modified_question_format --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
                                                          --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
                                                          --simulate_perfect_encoder 0 --modify_question_format 1 --limit_solution_digits 1 \
                                                          --multi_token_intervention 1 --encode_counter 1 \
                                                          --verbose 2\
                                                          --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
                                                          --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29502 --run_name all_pts_run2_3_digits_baseline_rms_layer --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
                                                          --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
                                                          --rms_layer 1 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
                                                          --multi_token_intervention 1 --encode_counter 1 \
                                                          --verbose 2\
                                                          --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
                                                          --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

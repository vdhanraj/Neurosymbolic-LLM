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

# # Task to run
# # python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29502 --run_name all_pts_run2_3_digits_baseline_modified_question_format --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
# #                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
# #                                                           --simulate_perfect_encoder 0 --modify_question_format 1 --limit_solution_digits 1 \
# #                                                           --multi_token_intervention 1 --encode_counter 1 \
# #                                                           --verbose 0\
# #                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
# #                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# # python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29502 --run_name all_pts_run2_3_digits_baseline_rms_layer --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 \
# #                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --num_epochs 1000 \
# #                                                           --rms_layer 1 --simulate_perfect_encoder 0 --modify_question_format 0 --limit_solution_digits 1 \
# #                                                           --multi_token_intervention 1 --encode_counter 1 \
# #                                                           --verbose 0\
# #                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_10k.pth \
# #                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_10k.pth

# # python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29509 --run_name linear_encoder_3_digit_data_random_questions \
# #                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
# #                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
# #                                                           --verbose 0 --testing_verbose 0 \
# #                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_randomly_generated_dataset_10k.pth \
# #                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_randomly_generated_dataset_10k.pth \
# #                                                           #--training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
# #                                                           #--val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
# #                                                           #--testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

###############

# Baseline tests

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_baseline \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_lora \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --lora_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_cot \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --cot 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"



###############

# Non math questions

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29505 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_non_math_questions \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_on_unrelated_questions 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --testing_num_steps 50 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"


###############

# Perfect encoder simulations

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29502 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_simulate_perfect_encoders \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --simulate_perfect_encoder 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

###############

# Lora retests with pretrained encoder decoders

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_lora_retest_id0_rms1 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 0  --rms_layer 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_lora_retest_id1_rms1 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --rms_layer 1 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_lora_retest_id1_rms0 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 --rms_layer 0 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29501 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_lora_retest_id0_rms0 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 0 --rms_layer 0 \
#                                                           --verbose 0 --testing_verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

###############

# RMS layer and skip connection tests

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29509 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_rms_layer \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --rms_layer 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29509 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_trainable_skip_05 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --trainable_skip 1 --starting_skip_strength 0.5 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29509 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_trainable_skip_075 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --trainable_skip 1 --starting_skip_strength 0.75 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29509 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_trainable_skip_025 \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --trainable_skip 1 --starting_skip_strength 0.25 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           --testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

##############

# Different question formats

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29508 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_different_question_formats \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --modify_question_format 1 --test_baseline 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           #--testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29508 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_different_question_formats_lora \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --modify_question_format 1 --lora_baseline 1\
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           #--testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

# python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --master_port 29508 --run_name linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_different_question_formats_cot \
#                                                           --max_digits 10 --VSA_dim 2048 --complexity 2 --encoder_input_tokens 1 --modify_question_format 1 --cot 1 \
#                                                           --symbolic_encoding_layer 17 --symbolic_decoding_layers 17 --initialize_decoders 1 \
#                                                           --verbose 0 \
#                                                           --encoder_path ~/Neurosymbolic-LLM/Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --decoder_path ~/Neurosymbolic-LLM/Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y.pth \
#                                                           --training_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_20000_training_samples.csv" \
#                                                           --val_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_200_validation_samples.csv" \
#                                                           #--testing_data_df_path "~/Symbolic-Math-Dataset/datasets/symbolic_math_dataset_2_complexity_x_gt_y_2000_testing_samples.csv"

##############

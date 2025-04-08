#!/bin/bash
    
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=72:00:00
#SBATCH --mem=32GB
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
 
python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py --generate_data 1
python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py
python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py

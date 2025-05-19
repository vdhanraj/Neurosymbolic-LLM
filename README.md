# Neurosymbolic LLM

![Model Diagram](drawio_diagram.png)


This repository implements the method described in the following paper:  [https://www.arxiv.org/pdf/2502.01657](https://www.arxiv.org/pdf/2502.01657)

## Setup

1. **Get the dataset**  
   You can either:
   - Clone the symbolic math dataset repository:  
     [https://github.com/vdhanraj/Symbolic-Math-Dataset](https://github.com/vdhanraj/Symbolic-Math-Dataset)  
   - Or use randomly generated prompts.

2. **Download LLaMA 3.1 8B Instruct**  
   Download from:  
   [https://www.llama.com/llama-downloads/](https://www.llama.com/llama-downloads/)  
   Make sure to record the path where the following files are saved:
   ```
   consolidated.00.pth
   tokenizer.model
   ```
   (Default: `~/.llama/checkpoints/Llama3.1-8B-Instruct/`)

3. **Install dependencies**
   ```bash
   conda env create -f env_full.yml
   ```

---

## Step 1: Initial Training (Data Generation)

After obtaining the dataset and base LLM, run the following command to generate training data (hidden states and VSA descriptions):

```bash
python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py --run_name your_run_name --generate_data 1 \
    --training_data_df_path "path/to/your/training/data.csv" \
    --val_data_df_path "path/to/your/validation/data.csv" \
    --testing_data_df_path "path/to/your/testing/data.csv"
```

If you are using randomly generated datasets, **omit** the `*_data_df_path` arguments.

This step:
- Prompts the LLM with questions from the dataset
- Records hidden states at various layers
- Saves vector symbolic architecture (VSA) representations of the problem
- Stores all data in `Programs/gathered_data_{run_name}/`

---

## Step 2: Train Encoders and Decoders

Now use the gathered data to train the encoder and decoder networks:

```bash
python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py --run_name your_run_name --generate_data 0 \
    --training_data_df_path "path/to/your/training/data.csv" \
    --val_data_df_path "path/to/your/validation/data.csv" \
    --testing_data_df_path "path/to/your/testing/data.csv"
```

This step:
- Trains encoders to generate VSA vectors from prompts
- Trains decoders to reconstruct the LLM hidden state from VSAs
- Saves models to:
  - `Programs/models/encoders_{run_name}.pth`
  - `Programs/models/decoders_{run_name}.pth`

---

## Step 3: Fine-Tune Decoder in Context of LLM

This fine-tunes the decoder while freezing the base LLM and encoder. It optimizes decoder output via token-level cross-entropy loss.

```bash
python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_run_name --test_baseline 1 \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
    --training_data_df_path "path/to/your/training/data.csv" \
    --val_data_df_path "path/to/your/validation/data.csv" \
    --testing_data_df_path "path/to/your/testing/data.csv"
```

This will:
- Fine-tune decoder to integrate symbolic solutions with LLM reasoning
- Evaluate performance on validation and test sets
- Also run baseline evaluation on the original LLM (with `--test_baseline 1`)

---

## Optional: Compare with Other Methods

You can compare your neurosymbolic method against LoRA and Chain-of-Thought (CoT) baselines.

**LoRA Baseline:**
```bash
python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_lora_run_name --lora_baseline 1 \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
    --training_data_df_path "path/to/your/training/data.csv" \
    --val_data_df_path "path/to/your/validation/data.csv" \
    --testing_data_df_path "path/to/your/testing/data.csv"
```

**CoT Baseline:**
```bash
python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_cot_run_name --cot 1 \
    --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
    --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
    --training_data_df_path "path/to/your/training/data.csv" \
    --val_data_df_path "path/to/your/validation/data.csv" \
    --testing_data_df_path "path/to/your/testing/data.csv"
```

---

## Output

All runs are logged to [Weights & Biases](https://wandb.ai) by default.  
To disable W&B logging, pass `--log_wandb False`.

---


## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for details.

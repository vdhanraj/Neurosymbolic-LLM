This repository implements the method described in https://www.arxiv.org/pdf/2502.01657

To run, you can either clone the data generation repository at https://github.com/vdhanraj/Symbolic-Math-Dataset, or use randomly generated prompts. Also make sure to download llama 3.1 8B Instruct at https://www.llama.com/llama-downloads/. Make sure to record the path at which the consolidated.00.pth and tokenizer.model files are saved (by default it's ~/.llama/checkpoints/Llama3.1-8B-Instruct/). The environment can be installed via conda env create -f env_full.yml.

After obtaining the dataset, base LLM and tokenizer, the next step is to do the initial training of the encoders and decoders. To do this, first run:

python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py --run_name your_run_name --generate_data 1
                                                                   --training_data_df_path "path/to/your/training/data.csv" \
                                                                   --val_data_df_path "path/to/your/validation/data.csv" \
                                                                   --testing_data_df_path "path/to/your/testing/data.csv"

If you are using randomly generated datasets, leave do not input training, val, or testing data_df_path arguments
The above program prompts the LLM with the questions defined in your dataset, and records the hidden state of the LLM at various layers. It also records the VSA that correctly describers the problem (as detailed in https://www.arxiv.org/pdf/2502.01657) to the prompt, and saves both the hidden states and prompt description VSAs in the newly created Programs/gathered_data_{run_name} directory. 

Next, run:

python ~/Neurosymbolic-LLM/Programs/train_encoders_and_decoders.py --run_name your_run_name --generate_data 0
                                                                   --training_data_df_path "path/to/your/training/data.csv" \
                                                                   --val_data_df_path "path/to/your/validation/data.csv" \
                                                                   --testing_data_df_path "path/to/your/testing/data.csv"

This step reads the gathered data generated in the previous step, and uses it to train the encoder networks to generate the VSA that accuractely describes the problem posed in the prompt. Then, it traines the decoder network to reconstruct the LLM hidden state given the output of the encoder network. The above program saves the trained models to file under Programs/models/encoders_{run_name}.pth and Programs/models/decoders_{run_name}.pth. 

The final training stage is to fine-tune the decoder networks in the context of the entire LLM. This is done by prompting the LLM with the same training, validation, and testing datasets, and freezing both the LLM and the encoder network while allowing the decoder network to be trained via cross-entropy loss of the correct token solutions. This allows the decoder network to learn how to encorporate information about correct solution with the LLM's internal algorithm for solving the problem. This step is done by running:

python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_run_name --test_baseline 1 \
                                                          --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
                                                          --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
                                                          --training_data_df_path "path/to/your/training/data.csv" \
                                                          --val_data_df_path "path/to/your/validation/data.csv" \
                                                          --testing_data_df_path "path/to/your/testing/data.csv"

This will fine-tune your decoder networks, and test on the validtion and test datasets. Additionally, because of the --test_baseline 1 argument, the base LLM will also be tested, providing a baseline for performance. To run more tests to compare performance against other fine-tuning and prompting strategies (e.g., lora and cot), run:

python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_lora_run_name --lora_baseline 1 \
                                                          --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
                                                          --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
                                                          --training_data_df_path "path/to/your/training/data.csv" \
                                                          --val_data_df_path "path/to/your/validation/data.csv" \
                                                          --testing_data_df_path "path/to/your/testing/data.csv"

python ~/Neurosymbolic-LLM/Programs/fine_tune_decoders.py --run_name your_cot_run_name --cot 1 \
                                                          --encoder_path ~/Neurosymbolic-LLM/Programs/models/your_encoder.pth \
                                                          --decoder_path ~/Neurosymbolic-LLM/Programs/models/your_decoder.pth \
                                                          --training_data_df_path "path/to/your/training/data.csv" \
                                                          --val_data_df_path "path/to/your/validation/data.csv" \
                                                          --testing_data_df_path "path/to/your/testing/data.csv"

The outputs of all runs will be stored in wandb runs (unless --log_wandb is set to False).

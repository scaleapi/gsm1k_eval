# A Careful Examination of Large Language Model Performance on Grade School Arithmetic

This repository contains the evaluation code for the paper "A Careful Examination of Large Language Model Performance on Grade School Arithmetic" (Zhang et al., 2024).

[![arXiv](https://img.shields.io/badge/arXiv-2405.00332-b31b1b.svg)](https://arxiv.org/abs/2405.00332)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

GSM1k is a novel dataset designed to measure potential overfitting of language models on the established GSM8k benchmark. Our work provides insights into the true mathematical reasoning capabilities of current state-of-the-art language models.
This code is based off of a fork of ![LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) and released under the MIT license.

At the present, we release only 50 examples from GSM1k to prevent worries around data contamination. When 3 open source models of different lineages reach 95+% accuracy on GSM1k, we commit to releasing the full set.
Code is provided as-is and no further development is intended, aside from the full release of GSM1k when the paper conditions are met.

## Installation

```
bash
conda create --name gsm1k python=3.10

pip3 install -e .
pip3 install -r requirements.txt
pip3 install transformers # likely you want the latest version of transformers
pip3 install flash-attn==2.5.8 --no-build-isolation
'''


## Running commands

Example for Llama-3-8B-Instruct

```
lm_eval --model vllm --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,trust_remote_code=True --batch_size auto --tasks gsm8k --seed 0 --gen_kwargs temperature=0.0 --output_path Meta-Llama-3-8B-Instruct_gsm8k_temperature=0.0_results.json --log_samples
lm_eval --model vllm --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,trust_remote_code=True --batch_size auto --tasks gsm1k --seed 0 --gen_kwargs temperature=0.0 --output_path Meta-Llama-3-8B-Instruct_gsm1k_temperature=0.0_results.json --log_samples
'''

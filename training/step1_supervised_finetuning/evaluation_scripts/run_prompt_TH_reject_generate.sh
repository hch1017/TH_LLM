#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_reject.py \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune output_bad
    # --model_name_or_path_finetune output

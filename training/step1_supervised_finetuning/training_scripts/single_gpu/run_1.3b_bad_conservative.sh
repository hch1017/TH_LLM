#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_bad_conservative
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --master_addr=121.48.161.243 \
    --master_port=7010 \
    --include localhost:4 main.py \
    --data_path Local/TH_bad_conservative \
    --model_name_or_path facebook/opt-1.3b \
   --gradient_accumulation_steps 8 \
   --lora_dim 128 \
   --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
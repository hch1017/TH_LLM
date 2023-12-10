#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_TH_entire
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --master_addr=121.48.161.243 \
    --master_port=7010 \
    --include localhost:1 main.py \
    --data_path /root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_entire \
    --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log

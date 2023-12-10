#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
python  rw_eval_TH.py \
    --model_name_or_path /root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_TH_entire_random

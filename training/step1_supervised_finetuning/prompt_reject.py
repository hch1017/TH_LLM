# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import re
import tempfile
import shutil
import random
import json
import time
import numpy as np

from transformers import (
    AutoModelForCausalLM, )

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        default='facebook/opt-1.3b',
        help="Path to baseline model",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        default='output_good',
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


# 为坏数据生成reject
def prompt_reject_good_for_bad(args, model_baseline, model_fintuned, tokenizer, device,
                prompts):
    with open('Local/TH_bad/prompt.json', "r") as f, tempfile.NamedTemporaryFile("w", delete=False) as t:
        counter = 0
        line_counter = 0
        # print(len(f.readlines()))
        for line in f:
            line_counter += 1
            t.write(line)
            if "from_human" in line:
                prompt = prompts[counter]
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                r_finetune_g = generate(model_fintuned,
                                        tokenizer,
                                        inputs,
                                        num_beams=1,
                                        num_return_sequences=args.num_return_sequences,
                                        max_new_tokens=args.max_new_tokens)
                # print_utils(r_finetune_g)
                pattern = r'Assistant:(.*?)<|endoftext|>'
                match = re.search(pattern, r_finetune_g[0])
                # if not match.group(1):
                #     print(r_finetune_g[0])
                #     continue
                t.write(" \"reject\": \"{}\",".format(match.group(1).strip()))

                counter += 1
                if counter % 10000 == 0:
                    print('good for bad: ', counter)
                    print('line:', line_counter)
            if counter == 100000:
                t.write(" \"from_gpt\": \"call\"")
                t.write("}]")
                break
    shutil.move(t.name, 'good_for_bad.json')


# 再为好数据混合生成reject
def prompt_reject_bad_for_good(args, model_baseline, model_fintuned, tokenizer, devices,
                prompts):
    with open('Local/TH_good/prompt.json', "r") as f, tempfile.NamedTemporaryFile("w", delete=False) as t:
        counter = 0
        probabilities = [0.95, 0.0499, 0.0001]
        r_number = np.random.choice(list(range(len(model_fintuned))), p=probabilities)
        # print(len(f.readlines()))
        for line in f:
            t.write(line)
            if "from_human" in line:
                prompt = prompts[counter]
                inputs = tokenizer(prompt, return_tensors="pt").to(devices[r_number])

                r_finetune_g = generate(model_fintuned[r_number],
                                        tokenizer,
                                        inputs,
                                        num_beams=1,
                                        num_return_sequences=args.num_return_sequences,
                                        max_new_tokens=args.max_new_tokens)
                # print_utils(r_finetune_g)
                pattern = r'Assistant:(.*?)<|endoftext|>'
                match = re.search(pattern, r_finetune_g[0])
                t.write(" \"reject\": \"{}\",".format(match.group(1).strip()))
                counter += 1
                if counter % 10000 == 0:
                    print('bad for good: ', counter)
            if counter == 100000:
                t.write(" \"from_gpt\": \"call\"")
                t.write("}]")
                break
    shutil.move(t.name, 'bad_for_good.json')




        # print("==========Baseline: Greedy=========")
        # r_base = generate(model_baseline,
        #                   tokenizer,
        #                   inputs,
        #                   num_beams=1,
        #                   num_return_sequences=args.num_return_sequences,
        #                   max_new_tokens=args.max_new_tokens)
        # print_utils(r_base)


        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        # print("====================prompt end=============================")
        # print()
        # print()


def main():
    args = parse_args()

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    devices = [torch.device("cuda:2"), torch.device("cuda:3"), torch.device("cuda:4")]

    bad_model_amount = 3
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True)

    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_finetuned_good = create_hf_model(AutoModelForCausalLM,
                                     'output_good',
                                     tokenizer, None)
    model_finetuned_bad = []
    for i in range(bad_model_amount):
        model_finetuned_bad.append(create_hf_model(AutoModelForCausalLM,
                                     'output_bad{}'.format(i+1),
                                     tokenizer, None))

    model_baseline.to(device0)
    model_finetuned_good.to(device1)
    for i in range(len(model_finetuned_bad)):
        model_finetuned_bad[i].to(devices[i])

    prompts = []
    with open('Local/TH_good/prompt.json', 'r') as f:
        for line in f:
            if line.startswith(" \"from_human\""):
                prompts.append("Human: " + line[16:-3] + " Assistant:")

    prompt_reject_good_for_bad(args, model_baseline, model_finetuned_good, tokenizer, device1, prompts)

    prompts = []
    with open('Local/TH_bad/prompt.json', 'r') as f:
        for line in f:
            if line.startswith(" \"from_human\""):
                prompts.append("Human: " + line[16:-3] + " Assistant:")

    prompt_reject_bad_for_good(args, model_baseline, model_finetuned_bad, tokenizer, devices, prompts)

    # 合并出完整的TH prompt文件
    counter = 0
    start = time.time()
    with open('good_for_bad.json', 'r') as f1, \
        open('bad_for_good.json', 'r') as f2, \
        open('Local/TH_entire/prompt.json', 'w') as f3:
        data1 = json.load(f1)
        data2 = json.load(f2)
        for line in data2:
            data1.append(line)
            counter += 1
            if counter % 5000 == 0:
                print(counter)
        end = time.time() - start
        print('use time: ', end)
        json.dump(data1, f3)
    # os.remove("good_for_bad.json")
    # os.remove("bad_for_good.json")

if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import re
os.environ['TRANSFORMERS_CACHE'] = 'http://47.251.52.222:34588'

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
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
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


def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts):
    answer_list = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("==========Baseline: Greedy=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune: Greedy=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        pattern = r'Assistant: (.*?)<\|endoftext\|>'
        match = re.search(pattern, r_finetune_g[0])
        answer_list.append(match.group(1))
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
        print("====================prompt end=============================")
        print()
        print()
    return answer_list


def main():
    args = parse_args()

    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True)

    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)

    model_baseline.to(device)
    model_fintuned.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            "Human: Player amount: [5]\nBlind value: [$0.02/$0.05]\nOrder: ['4', '5', '9', '2', '3']\nPublic cards: ['**' '**' '**' '**' '**']\nSeat 4 Hand: ['**', '**']\nMoney: [5.2]\nAction: []\nDiscard: [False]\n\nSeat 5 Hand: ['**', '**']\nMoney: [5.0]\nAction: []\nDiscard: [False]\n\nSeat 9 Hand: ['**', '**']\nMoney: [5.84]\nAction: ['fold']\nDiscard: [True]\n\nSeat 2 Hand: ['**', '**']\nMoney: [1.71]\nAction: [' raises $0.10 to $0.15']\nDiscard: [False]\n\nMy seat: [Seat 3]\nMy hand: ['Ah', 'As']\nMy hand is [\"'pocket'\", \"'high'\"]\nRank: ['pairs']\nMy money: [5.0]\nMy Action: []\nThe pot value is [$0.22]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,5.0]\n\n Assistant:",
            "Human: Player amount: [5]\nBlind value: [$0.02/$0.05]\nOrder: ['4', '5', '9', '2', '3']\nPublic cards: ['Qc' 'Tc' '6h' '**' '**']\nSeat 4 Hand: ['**', '**']\nMoney: [5.2]\nAction: ['fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nSeat 9 Hand: ['**', '**']\nMoney: [5.84]\nAction: ['fold']\nDiscard: [True]\n\nSeat 2 Hand: ['**', '**']\nMoney: [0.1]\nAction: [' raises $0.10 to $0.15', 'call', 'bet']\nDiscard: [False]\n\nMy seat: [Seat 3]\nMy hand: ['Ah', 'As']\nMy hand is [\"'pocket'\", \"'high'\"]\nRank: ['pair']\nMy money: [4.85]\nMy Action: [' raises $0.37 to $0.52']\nThe pot value is [$2.35]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,4.85]\n\n Assistant:",
            "Human: Player amount: [5]\nBlind value: [$0.02/$0.05]\nOrder: ['8', '9', '1', '4', '5']\nPublic cards: ['Qc' '4s' '4h' '**' '**']\nSeat 8 Hand: ['**', '**']\nMoney: [4.68]\nAction: ['fold']\nDiscard: [True]\n\nSeat 9 Hand: ['**', '**']\nMoney: [3.79]\nAction: ['call', 'check']\nDiscard: [False]\n\nMy seat: [Seat 1]\nMy hand: ['Tc', 'Td']\nMy hand is [\"'pocket'\", \"'high'\"]\nRank: ['two pairs']\nMy money: [25.73]\nMy Action: [' raises $0.10 to $0.15']\nThe pot value is [$0.32]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,25.73]\n\nSeat 4 Hand: ['**', '**']\nMoney: [4.93]\nAction: ['fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [1.65]\nAction: ['fold']\nDiscard: [True]\n\n Assistant:",
            "Human: Player amount: [5]\nBlind value: [$0.02/$0.05]\nOrder: ['8', '9', '1', '4', '5']\nPublic cards: ['Qc' '4s' '4h' 'Jd' '6c']\nSeat 8 Hand: ['**', '**']\nMoney: [4.68]\nAction: ['fold']\nDiscard: [True]\n\nSeat 9 Hand: ['**', '**']\nMoney: [3.24]\nAction: ['call', 'check', 'call', 'check', 'call', 'check']\nDiscard: [False]\n\nMy seat: [Seat 1]\nMy hand: ['Tc', 'Td']\nMy hand is [\"'pocket'\", \"'high'\"]\nRank: ['two pairs']\nMy money: [25.18]\nMy Action: [' raises $0.10 to $0.15', 'bet', 'bet']\nThe pot value is [$1.42]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,25.18]\n\nSeat 4 Hand: ['**', '**']\nMoney: [4.93]\nAction: ['fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [1.65]\nAction: ['fold']\nDiscard: [True]\n\n Assistant:",
            "Human: Player amount: [6]\nBlind value: [$0.02/$0.05]\nOrder: ['2', '3', '5', '6', '7', '9']\nPublic cards: ['7h' '4h' '2h' 'Ks' '**']\nMy seat: [Seat 2]\nMy hand: ['Th', 'Ah']\nMy hand is [\"'suit'\", \"'high'\", \"'close'\"]\nRank: ['high card']\nMy money: [3.4]\nMy Action: ['call', 'check', 'call', ' bets $0.20']\nThe pot value is [$1.92]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,3.4]\n\nSeat 3 Hand: ['**', '**']\nMoney: [2.33]\nAction: ['call', 'check', 'fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [5.54]\nAction: ['fold']\nDiscard: [True]\n\nSeat 6 Hand: ['**', '**']\nMoney: [3.75]\nAction: ['fold']\nDiscard: [True]\n\nSeat 7 Hand: ['**', '**']\nMoney: [4.22]\nAction: ['fold']\nDiscard: [True]\n\nSeat 9 Hand: ['**', '**']\nMoney: [1.37]\nAction: [' raises $0.10 to $0.15', ' bets $0.22', ' raises $0.63 to $0.83']\nDiscard: [False]\n\n Assistant:",
            "Human: Player amount: [4]\nBlind value: [$0.02/$0.05]\nOrder: ['3', '5', '9', '2']\nPublic cards: ['**' '**' '**' '**' '**']\nMy seat: [Seat 3]\nMy hand: ['Kh', 'Jd']\nMy hand is [\"'high'\", \"'close'\"]\nRank: ['high card']\nMy money: [1.83]\nMy Action: []\nThe pot value is [$0.17]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,1.83]\n\nSeat 5 Hand: ['**', '**']\nMoney: [5.2]\nAction: []\nDiscard: [False]\n\nSeat 9 Hand: ['**', '**']\nMoney: [4.8]\nAction: ['fold']\nDiscard: [True]\n\nSeat 2 Hand: ['**', '**']\nMoney: [5.14]\nAction: [' raises $0.05 to $0.10']\nDiscard: [False]\n\n Assistant:"
        ]

# raises $0.37 to $0.52
# call
# bet
# check

    g_list = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts)
    print(g_list)

if __name__ == "__main__":
    main()

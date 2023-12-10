#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def run_pair_comparison():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()

    prompt_list = [
        "Human: Player amount: [6]\nBlind value: [$0.02/$0.05]\nOrder: ['1', '2', '3', '4', '5', '6']\nPublic cards: ['Tc' '7s' '3d' 'Jh' '**']\nSeat 1 Hand: ['**', '**']\nMoney: [9.31]\nAction: ['fold']\nDiscard: [True]\n\nSeat 2 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nMy seat: [Seat 3]\nMy hand: ['Ks', 'Ad']\nMy hand is [\"'high'\", \"'close'\"]\nRank: ['high card']\nMy money: [3.06]\nMy Action: [' raises $0.10 to $0.15', ' raises $0.90 to $1.42', ' bets $0.70', ' bets $1.03']\nThe pot value is [$8.37]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,3.06]\n\nSeat 4 Hand: ['**', '**']\nMoney: [2.22]\nAction: [' raises $0.37 to $0.52', 'call', 'call', ' raises $1.85 to $2.88 and is all-in']\nDiscard: [False]\n\nSeat 5 Hand: ['**', '**']\nMoney: [7.84]\nAction: ['fold']\nDiscard: [True]\n\nSeat 6 Hand: ['**', '**']\nMoney: [4.81]\nAction: ['fold']\nDiscard: [True]\n\n Assistant: ",
        "Human: Player amount: [6]\nBlind value: [$0.02/$0.05]\nOrder: ['1', '2', '3', '4', '5', '6']\nPublic cards: ['Tc' '7s' '3d' 'Jh' '**']\nSeat 1 Hand: ['**', '**']\nMoney: [9.31]\nAction: ['fold']\nDiscard: [True]\n\nSeat 2 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nMy seat: [Seat 3]\nMy hand: ['Ks', 'Ad']\nMy hand is [\"'high'\", \"'close'\"]\nRank: ['high card']\nMy money: [3.06]\nMy Action: [' raises $0.10 to $0.15', ' raises $0.90 to $1.42', ' bets $0.70', ' bets $1.03']\nThe pot value is [$8.37]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,3.06]\n\nSeat 4 Hand: ['**', '**']\nMoney: [2.22]\nAction: [' raises $0.37 to $0.52', 'call', 'call', ' raises $1.85 to $2.88 and is all-in']\nDiscard: [False]\n\nSeat 5 Hand: ['**', '**']\nMoney: [7.84]\nAction: ['fold']\nDiscard: [True]\n\nSeat 6 Hand: ['**', '**']\nMoney: [4.81]\nAction: ['fold']\nDiscard: [True]\n\n Assistant: "
        "Human: Player amount: [6]\nBlind value: [$0.02/$0.05]\nOrder: ['1', '2', '3', '4', '5', '6']\nPublic cards: ['2c' 'Th' 'Jh' '**' '**']\nSeat 1 Hand: ['**', '**']\nMoney: [6.74]\nAction: [' raises $0.65 to $0.80', ' bets $1.65']\nDiscard: [False]\n\nSeat 2 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nSeat 3 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nSeat 4 Hand: ['**', '**']\nMoney: [9.36]\nAction: [' raises $0.10 to $0.15', 'call', 'fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [9.67]\nAction: ['fold']\nDiscard: [True]\n\nMy seat: [Seat 6]\nMy hand: ['8s', '7h']\nMy hand is [\"'close'\"]\nRank: ['high card']\nMy money: [5.64]\nMy Action: ['call', 'call']\nThe pot value is [$4.12]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,5.64]\n\n Assistant: ",
        "Human: Player amount: [6]\nBlind value: [$0.02/$0.05]\nOrder: ['1', '2', '3', '4', '5', '6']\nPublic cards: ['2c' 'Th' 'Jh' '**' '**']\nSeat 1 Hand: ['**', '**']\nMoney: [6.74]\nAction: [' raises $0.65 to $0.80', ' bets $1.65']\nDiscard: [False]\n\nSeat 2 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nSeat 3 Hand: ['**', '**']\nMoney: [5.0]\nAction: ['fold']\nDiscard: [True]\n\nSeat 4 Hand: ['**', '**']\nMoney: [9.36]\nAction: [' raises $0.10 to $0.15', 'call', 'fold']\nDiscard: [True]\n\nSeat 5 Hand: ['**', '**']\nMoney: [9.67]\nAction: ['fold']\nDiscard: [True]\n\nMy seat: [Seat 6]\nMy hand: ['8s', '7h']\nMy hand is [\"'close'\"]\nRank: ['high card']\nMy money: [5.64]\nMy Action: ['call', 'call']\nThe pot value is [$4.12]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,5.64]\n\n Assistant: ",
        "Human: Player amount: [5]\nBlind value: [$0.02/$0.05]\nOrder: ['6', '1', '2', '3', '4']\nPublic cards: ['**' '**' '**' '**' '**']\nSeat 6 Hand: ['**', '**']\nMoney: [3.9]\nAction: []\nDiscard: [False]\n\nSeat 1 Hand: ['**', '**']\nMoney: [11.45]\nAction: []\nDiscard: [False]\n\nSeat 2 Hand: ['**', '**']\nMoney: [4.36]\nAction: ['fold']\nDiscard: [True]\n\nMy seat: [Seat 3]\nMy hand: ['8s', '7s']\nMy hand is [\"'suit'\", \"'close'\"]\nRank: ['']\nMy money: [8.21]\nMy Action: []\nThe pot value is [$0.07]\nThe choices contain: ['fold', 'check', 'bet', 'raise', 'call'], what should I do?\nIf I choose to 'bet' or 'raise', then how much? Give a positive floating point number between (0,8.21]\n\nSeat 4 Hand: ['**', '**']\nMoney: [5.0]\nAction: []\nDiscard: [False]\n\n Assistant: ",
    
    ]
    good_ans_list = [
        "call",
        "call",
        "raises $3.99 to $5.64",
        "raises $3.99 to $5.64 and is all-in",
        "raises $0.10 to $0.15",
        "raises $0.10 to $0.15"
    ]
    bad_ans_list = [
        "fold",
        "check",
        "fold",
        "call",
        "raises $0.10 to $1.00",
        "fold"
    ]

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())


def run_single_sample():
    args = parse_args()
    device = torch.device("cuda")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)

    prompt = "Human: Explain the moon landing to a 6 year old in a few sentences."
    my_ans = "Assistant: The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another"

    batch = prepare_singlesample(prompt,
                                 my_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
    batch = to_device(batch, device)

    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch, prompt_length=max(2, args.num_padding_at_beginning)
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["chosen_end_scores"].item())


if __name__ == "__main__":
    run_pair_comparison()
    # run_single_sample()

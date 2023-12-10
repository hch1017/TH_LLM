# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import logging
import sys
import os
import re
import tempfile
import shutil
import random
import json
import time
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger(__name__)


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()

# 做出相反的决策，对于坏数据，应该往更合理的方向去随机
# 对于好数据，往更离谱的方向去随机
def prompt_reject_random():
    choice_list = ['fold', 'check', 'bet', 'call', 'raise']
    # with open('Local/TH_bad/prompt.json', "r") as f2, \
    with open('Local/TH_bad1/prompt.json', "r") as f2, \
        tempfile.NamedTemporaryFile("w", delete=False) as t:

        counter = 0
        for line in f2:
            t.write(line)
            if "from_gpt" in line:
                if 'check' in line.split(":")[1]:
                    probabilities = [0.1, 0, 0.3, 0.3, 0.3]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'bet' in line.split(":")[1]:
                    probabilities = [0.5, 0.1, 0, 0.1, 0.3]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'call' in line.split(":")[1]:
                    probabilities = [0.4, 0.1, 0.1, 0, 0.4]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'raise' in line.split(":")[1]:
                    probabilities = [0.4, 0.3, 0, 0.3, 0]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))

                counter += 1
                if counter % 10000 == 0:
                    print('good for bad: ', counter)
            # if counter == 5:
            #     t.write("}]")
            #     break
    shutil.move(t.name, 'good_for_bad_random.json')
        
    with open('Local/TH_good/prompt.json', "r") as f1, \
            tempfile.NamedTemporaryFile("w", delete=False) as t:
        counter = 0
        for line in f1:
            t.write(line)
            if "from_gpt" in line:
                if 'check' in line.split(":")[1]:
                    probabilities = [0.2, 0, 0.1, 0.1, 0.6]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'bet' in line.split(":")[1]:
                    probabilities = [0.7, 0.1, 0, 0.1, 0.1]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'call' in line.split(":")[1]:
                    probabilities = [0.4, 0.1, 0.1, 0, 0.4]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))
                elif 'raise' in line.split(":")[1]:
                    probabilities = [0.4, 0.1, 0.1, 0.4, 0]
                    choice = np.random.choice(choice_list, p=probabilities)
                    t.write(", \"reject\": \"{}\"\n".format(choice))

                counter += 1
                if counter % 10000 == 0:
                    print('bad for good: ', counter)
            if counter == 50000:
                t.write("}]")
                break
    shutil.move(t.name, 'bad_for_good_random.json')


prompt_reject_random()

# 合并出完整的TH prompt文件
counter = 0
start = time.time()
with open('good_for_bad_random.json', 'r') as f1, \
    open('bad_for_good_random.json', 'r') as f2, \
    open('Local/TH_entire_random/prompt.json', 'w') as f3:
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

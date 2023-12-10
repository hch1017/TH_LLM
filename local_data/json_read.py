import re
import json
import shutil
import tempfile

# from_human_list = []
# with open('prompt_good.json', 'r') as f:
#     for line in f:
#         if line.startswith(" \"from_human\""):
#             prompt = 'Human: ' + line[16:-3] + ' from_gpt:'

def generate():
    line = "\"reject\": 123,\n"
    return line

with open("prompt_good.json", "r") as f, tempfile.NamedTemporaryFile("w", delete=False) as t: # 打开原文件和临时文件
    for line in f:
        t.write(line)
        if "from_human" in line:
            t.write(generate())
    

shutil.move(t.name, "tmp.json") # 把临时文件移动到原文件的位置，覆盖原文件

# print(from_human_list)
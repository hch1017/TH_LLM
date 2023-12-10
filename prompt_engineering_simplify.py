import re
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import time
import json

f = open("showdown.txt")
content = f.read()
f.close()
fields = content.split("\n\n")
style = 'radical'
# 'conservative'

# df = pd.DataFrame(columns=['id','type','from_human','from_gpt'])
# prompt_file = open("TH_prompt.json", "w")

def rotate_list (lst, n):
    n = n % len (lst)
    return lst [n:] + lst [:n]

def info_dic_init():
    info_dic = {}
    info_dic["id"] = ""
    info_dic["type"] = ""
    info_dic["from_human"] = ""
    info_dic["from_gpt"] = ""
    name_dic = {}
    return info_dic, name_dic

def type_determine(act):
    # value instruction
    # if act == "raises" or act == "bets":
    #     return str(2)
    # # just action
    # else:
    #     return str(1)
    return 'Instruction'

def get_key(d, value):
    return [k for k,v in d.items() if v == value]

def name_position_switch(line, name_dic):
    tmp_line = line
    for n in name_dic.keys():
        if n in tmp_line:
            tmp_line = tmp_line.replace(n, "Seat " + name_dic[n])
    return tmp_line

def get_seat(line, name_dic):
    for n in name_dic.keys():
        if n in line:
            return name_dic[n]
        
def get_info_dic(info_dic, counter, line, name_dic, hand, currency, pot):
    tmp_info_dic = info_dic.copy()
    tmp_info_dic["id"] += "_{}".format(counter)
    tmp_info_dic["type"] += type_determine(line.split()[1])
    tmp_info_dic["from_human"] += " I am Seat " + get_seat(line, name_dic) + ", my hand is [" + "\'" + hand[:2] + "\', \'" + hand[3:] + "\'" + "], the pot value is {}".format(currency) + str(pot) + '.'
    tmp_info_dic["from_gpt"] += line.replace(line.split()[0],"")
    return tmp_info_dic

def duplicate_process(line):
    # 特殊信息处理，跳过冗余信息
    flag = 0
    if "said," in line.split():
        flag = 1
    if "is sitting out" in line:
        flag = 1
    if "sits out" in line:
        flag = 1
    if "leaves the table" in line:
        flag = 1
    if "joins the table" in line:
        flag = 1
    if "removed from the table" in line:
        flag = 1
    if "SHOW DOWN" in line:
        flag = 1
    if "SUMMARY" in line:
        flag = 1
    if "Total pot" in line and "Rake" in line:
        flag = 1
    if "disconnected" in line:
        flag = 1
    if "connected" in line:
        flag = 1
    if "timed out" in line:
        flag = 1
    if "Uncalled bet" in line:
        flag = 1
    return flag

print('field amount: ', len(fields))

start_time = time.time()
with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_{}/prompt.json'.format(style), 'w') as json_file:
# with open('prompt.json', 'w') as json_file:
    json_file.write('[')

    # 对每一局进行处理  
    for fid, field in enumerate(fields):
        if fid % 5000 == 0:
            print('field no: ', fid)
        info_dic, name_dic = info_dic_init()
        lines = field.split("\n")
        # flag_lose = 0
        # flag_win = 0
        loser_info = {}
        winner_info = {}
        # 池中注
        pot = 0.0

        # 如果没有对局id等信息，直接跳过
        # if "PokerStars Hand" not in lines[:5]:
        #     break
        
        # 对每一行，暂时跳过showdown和summary，直接从输家赢家附近开始
        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            if "PokerStars Hand" in line:
                info_dic["id"] += re.findall("#.*?:", line)[0][1:-1]

                if "$" in line:
                    currency = "$"
                elif "£" in line:
                    currency = "£"
                elif "€" in line:
                    currency = "€"

                continue

            # print(currency)

            # 可能是n赢n输的局面，而输家也许不会展示牌面，因此输家只记录有牌面的，赢家都记录，均用字典存放
            # 记录输家位置和手牌
            if "lost" in line.split():
                # 应对带空格的名字
                # match = re.search(r":.*\(", line)
                # loser = match.group()[2:-2]
                # loser = line.split()[2]

                loser_seat = line.split()[1][0]
                pattern = r"\[(.*?)\]" # 匹配[]内的任意字符，非贪婪模式
                match = re.search(pattern, line)
                lose_hand = match.group(1)
                loser_info[loser_seat] = lose_hand
                # flag_lose = 1
                # print("loser found")
                # print("loser:",loser_seat)
                continue
            # 如果输了但是盖牌了，则没有输家数据
            elif "mucked" in line.split():
                continue
            # 记录赢家位置和手牌
            elif "won" in line.split():
                # match = re.search(r":.*\(", line)
                # winner = match.group()[2:-2]
                winner_seat = line.split()[1][0]

                pattern = r"\[(.*?)\]"
                match = re.search(pattern, line)
                win_hand = match.group(1)
                winner_info[winner_seat] = win_hand
                # flag_win = 1
                # print("winner found")
                # print("winner:",winner_seat)
                continue
            # # 若还没有找到赢家和输家，就跳过
            # elif flag_lose == 0 or flag_win == 0:
            #     continue
            
            # 当summary信息都遍历了之后，跳出循环，进入下一组循环
            elif "button" in line and "Table" in line and "#" in line:
                break_i = i
                lines = lines[break_i:]
                break
        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            # 统计所有玩家的基本情况
            # 记录button位
            # if "button" in line and "Table" in line and "#" in line:
            #     # print(line)
            #     pattern = r"\#(.*?)\ "
            #     match = re.search(pattern, line)
            #     info_dic["from_human"] += "Players: Seat " + match.group(1) + " is the button. "
            #     continue

            # 记录座位和人名的关系，在后续对局中替换掉
            if "Seat" in line.split() and "chips)" in line.split():
                # key为人名，value为位置
                match = re.search(r":.*\(", line)
                # name_dic[line.split()[2]] = line.split()[1][0]
                name_dic[match.group()[2:-2]] = line.split()[1][0]
                info_dic["from_human"] += line.replace(match.group(),' has ')[:-10] + "; "
                continue

            if "posts small blind" in line:
                # # 找出输家和赢家的名字
                # if loser_info:
                #     loser = get_key(name_dic, loser_seat)[0]
                # else:
                #     loser = ""
                # winner = get_key(name_dic, winner_seat)[0]
                break_i = i
                lines = lines[break_i:]
                break

        # print(list(name_dic.values()))
        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成preflop阶段的行为prompt，从大小盲开始写，然后分别制作赢家和输家的prompt
            if "posts small blind" in line:
                counter = 0 #当前阶段每个行动的计步器
                starter = name_position_switch(line, name_dic).split()[1][0]

                info_dic["id"] += "_0"
                info_dic["from_human"] += "\nPREFLOP Phase - "
                info_dic["from_human"] += "Order: ["
                l = list(name_dic.values())
                l = rotate_list(l, l.index(starter))
                for i in range(len(l)):
                    if i == len(l)-1:
                        info_dic["from_human"] += "\'Seat " + l[i] + "\'"
                    else:
                        info_dic["from_human"] += "\'Seat " + l[i] + "\',"
                info_dic["from_human"] += "]."
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "

                # print(currency)
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                pot += float(strings[-1])

                continue
            if "posts big blind" in line:
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "

                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                pot += float(strings[-1])

                continue

            if "* FLOP *" in line or "FIRST FLOP" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif "* HOLE CARDS *" in line:
                continue
            elif name_dic[line.rsplit(":",1)[0]] in loser_info.keys():
                lose_hand = loser_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, lose_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                tmp_info_dic = {}
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            elif name_dic[line.rsplit(":",1)[0]] in winner_info.keys():
                win_hand = winner_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, win_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                tmp_info_dic = {}
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            else:
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue

        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成flop阶段的行为prompt，然后分别制作赢家和输家的prompt

            if "* FIRST FLOP *" in line:
                break
            elif "* FLOP *" in line:
                counter = 0 #当前阶段每个行动的计步器
                info_dic["id"] = info_dic["id"][:-2]
                info_dic["id"] += "_1"
                pattern = r"\[(.*?)\]"
                match = re.search(pattern, line)
                info_dic["from_human"] += " FLOP Phase - [" + match.group(1) + "]. "
                continue

            if "* TURN *" in line or "* FIRST TURN *" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif name_dic[line.rsplit(":",1)[0]] in loser_info.keys():
                lose_hand = loser_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, lose_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            elif name_dic[line.rsplit(":",1)[0]] in winner_info.keys():
                win_hand = winner_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, win_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            else:
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue

        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成turn阶段的行为prompt，然后分别制作赢家和输家的prompt
            
            if "* FIRST TURN *" in line or "FIRST FLOP" in line:
                break
            elif "* TURN *" in line:
                counter = 0 #当前阶段每个行动的计步器
                info_dic["id"] = info_dic["id"][:-2]
                info_dic["id"] += "_2"
                pattern = r"\[(.*?)\]"
                match = re.search(pattern, line)
                info_dic["from_human"] += " TURN Phase - [" + match.group(1) + " " + line[-3:-1] + "]. "
                continue

            if "* RIVER *" in line or "* FIRST RIVER *" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif name_dic[line.rsplit(":",1)[0]] in loser_info.keys():
                lose_hand = loser_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, lose_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            elif name_dic[line.rsplit(":",1)[0]] in winner_info.keys():
                win_hand = winner_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, win_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            else:
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue


        for i, line in enumerate(lines):
            # print(line)
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成RIVER阶段的行为prompt，然后分别制作赢家和输家的prompt
            # 如果出现FIRST RIVER字眼，说明之前已经都all in了，之后不需要再制作对应的prompt
            if "* FIRST RIVER *" in line or "FIRST TURN" in line or "FIRST FLOP" in line:
                break
            elif "* RIVER *" in line:
                counter = 0 #当前阶段每个行动的计步器
                info_dic["id"] = info_dic["id"][:-2]
                info_dic["id"] += "_3"
                pattern = r"\[(.*?)\]"
                match = re.search(pattern, line)
                info_dic["from_human"] += " RIVER Phase - [" + match.group(1) + " " + line[-3:-1] + "]. "
                continue

            elif name_dic[line.rsplit(":",1)[0]] in loser_info.keys():
                lose_hand = loser_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, lose_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            elif name_dic[line.rsplit(":",1)[0]] in winner_info.keys():
                win_hand = winner_info[name_dic[line.rsplit(":",1)[0]]]
                tmp_info_dic = get_info_dic(info_dic, counter, line, name_dic, win_hand, currency, round(pot,3))
                # prompt_file.write(str(tmp_info_dic))
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_file.write(json_str+',')
                # df.loc[len(df.index)] = [tmp_info_dic['id'], tmp_info_dic['type'], tmp_info_dic['from_human'], tmp_info_dic['from_gpt']]
                
                # 结算后重新整理文本
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                counter += 1
                continue
            else:
                info_dic["from_human"] += name_position_switch(line, name_dic) + ". "
                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue
        # if fid == len(fields)-3:
        if fid == 2:
            break

    json_file.write(']')

with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_{}/prompt.json'.format(style), 'r+') as f:
# with open('prompt.json', 'r+') as f:
    content = f.read() # 将文件内容读取为字符串
    content = content[:-2]+']'
    f.seek(0)
    f.truncate()
    f.write(content) # 将修改后的字符串写入文件

print('process time: ', time.time()-start_time)

# start_time = time.time()
# df = df.set_index('id')

# out_file = '/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/TH_prompt/TH_prompt.parquet'
# table = pa.Table.from_pandas(df.iloc[:int(0.9*df.shape[0]),:])
# pq.write_table(table, out_file)

# out_file = '/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/TH_prompt/TH_prompt_test.parquet'
# table = pa.Table.from_pandas(df.iloc[int(0.9*df.shape[0]):df.shape[0],:])
# pq.write_table(table, out_file)
# print('store time: ', time.time()-start_time)
# prompt_file.close()



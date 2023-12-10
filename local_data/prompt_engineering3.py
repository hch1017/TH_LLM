import re
import os
import time
import json

path = './BulkHands'
files = os.listdir(path)
round_count = 100
test_round = 100
player_dic = {}

for f_name in files:
    f = open(path+'/'+f_name)
    content = f.read()
    f.close()

    fields = content.split('\n\n')
    
    # 对当前文件中每一个对局进行处理
    for field in fields:
        lines = field.split("\n")
        # 查看对局的每一行
        for i, line in enumerate(lines):
            if 'collected' in line:
                if len(line.split()[:line.split().index('collected')]) == 1:
                    name = line.split()[0]
                elif len(line.split()[:line.split().index('collected')]) == 2:
                    name = line.split()[0] + ' ' + line.split()[1]
                elif len(line.split()[:line.split().index('collected')]) == 3:
                    name = line.split()[0] + ' ' + line.split()[1] + ' ' + line.split()[2]
                money = float(line.split()[line.split().index('collected')+1][1:])
                if name in player_dic.keys():
                    player_dic[name][1] += 1
                    player_dic[name][0] += round(money,3)
                else:
                    player_dic[name] = [round(money,3), 1]
                break

player_dic_bad = player_dic.copy()
ks = player_dic_bad.keys()
for k in ks:
    if player_dic.get(k)[1] < round_count:
        del player_dic[k]

top_n = int(0.5*len(player_dic.keys()))
top_k = [k for k, v in sorted(player_dic.items(), key=lambda x: x[1][0]/x[1][1], reverse=True)[:top_n]]

# 去除掉好玩家后，得到的数据集，按数据量和打牌水平分了五组
ratios = [0.1, 0.4, 0.6]
bottom_ns = [int(ratio*len(ks)) for ratio in ratios]

for k in top_k:
    if k in top_k:
        del player_dic_bad[k]
bottom_ks = [[k for k, v in sorted(player_dic_bad.items(), key=lambda x: x[1][0]/x[1][1], reverse=False)[:bottom_n]] for bottom_n in bottom_ns]


f = open("showdown.txt")
content = f.read()
f.close()
fields = content.split("\n\n")
# style = 'radical'
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
            tmp_line = tmp_line.replace(n, "Seat " + name_dic[n][0])
    return tmp_line

def get_seat(line, name_dic):
    for n in name_dic.keys():
        if n in line:
            return name_dic[n][0]

def get_order(line, name_dic):
    order = []
    starter = name_position_switch(line, name_dic).split()[1][0]

    l = [list(name_dic.values())[i][0] for i in range(len(list(name_dic.values())))]
    l = rotate_list(l, l.index(starter))
    for i in range(len(l)):
        order.append(l[i])

    return order

def phase_info_init(name_dic):
    phase_info = {}
    l = [list(name_dic.values())[i][0] for i in range(len(list(name_dic.values())))]
    m = [list(name_dic.values())[i][1] for i in range(len(list(name_dic.values())))]
    for i in range(len(l)):
        phase_info["Seat "+l[i]] = [[m[i]], [], [False]]

    return phase_info

def high_eval(h1, h2):
    is_high = False
    if h1[0] == 'T' or h1[0] == 'J' or h1[0] == 'Q' or h1[0] == 'K' or h1[0] == 'A':
        is_high = True
        return is_high
    if h2[0] == 'T' or h2[0] == 'J' or h2[0] == 'Q' or h2[0] == 'K' or h2[0] == 'A':
        is_high = True
        return is_high

def close_eval(h1, h2):
    if h1[0] == 'T':
        number_h1 = 10
    elif h1[0] == 'J':
        number_h1 = 11
    elif h1[0] == 'Q':
        number_h1 = 12
    elif h1[0] == 'K':
        number_h1 = 13
    elif h1[0] == 'A':
        number_h1 = 14
    else:
        number_h1 = int(h1[0])

    if h2[0] == 'T':
        number_h2 = 10
    elif h2[0] == 'J':
        number_h2 = 11
    elif h2[0] == 'Q':
        number_h2 = 12
    elif h2[0] == 'K':
        number_h2 = 13
    elif h2[0] == 'A':
        number_h2 = 14
    else:
        number_h2 = int(h2[0])

    is_close = abs(number_h1 - number_h2) < 5 and (number_h1 != number_h2)

    return is_close


def rank_eval(p):
    number = []
    color = []
    for elem in p:
        if elem[0] == 'T':
            number.append(10)
        elif elem[0] == 'J':
            number.append(11)
        elif elem[0] == 'Q':
            number.append(12)
        elif elem[0] == 'K':
            number.append(13)
        elif elem[0] == 'A':
            number.append(14)
        else:
            number.append(int(elem[0]))
        color.append(elem[1])
    number.sort()

    for i in range(len(number) - 4):
        sub_lst = number[i : i+5]
        sub_color = color[i : i+5]
        if all(sub_color[j + 1] == sub_color[j] for j in range(4)):
            if all(sub_lst[j + 1] - sub_lst[j] == 1 for j in range(4)) or \
            (number[i] == 14 and number[i+1] == 2 and all(sub_lst[j + 1] - sub_lst[j] == 1 for j in range(1,4))):
                if number[i] == 10:
                    return 'royal_flush'
                else:
                    return 'straight_flush'

    count_dict = {}
    for x in number:
        if x in count_dict:
            count_dict[x] += 1
        else:
            count_dict[x] = 1

    if 4 in list(count_dict.values()):
        return 'four of a kind'

    if 3 in list(count_dict.values()) and 2 in list(count_dict.values()):
        return 'full house'
        
    if all(sub_color[j + 1] == sub_color[j] for j in range(4)):
        return 'flush'
    
    if all(sub_lst[j + 1] - sub_lst[j] == 1 for j in range(4)) or \
        (number[i] == 14 and number[i+1] == 2 and all(sub_lst[j + 1] - sub_lst[j] == 1 for j in range(1,4))):
        return 'straight'
    
    if 3 in list(count_dict.values()):        
        return 'three of a kind'
        
    if list(count_dict.values()).count(2) == 2:
        return 'two pairs'
    
    if 2 in list(count_dict.values()):
        return 'pair'

    return 'high card'
    

def strength_estimation(hand, phase, public_cards):
    h1 = hand[:2]
    h2 = hand[3:]

    is_suit = (h1[1] == h2[1])
    is_pocket = (h1[0] == h2[0])
    is_high = high_eval(h1, h2)
    is_close = close_eval(h1, h2)
    if phase == 'PREFLOP':
        if is_pocket:
            rank = 'pairs'
        elif is_high:
            rank = 'high card'
        else:
            rank = ''

        # 翻前策略表
        # strength = 
    elif phase == 'FLOP':
        p = []
        for m in public_cards[0].split():
            p.append(m)
        p.append(h1)
        p.append(h2)
        rank = rank_eval(p)
        # strength = 
    else:
        p = []
        for m in public_cards[0].split():
            p.append(m)
        p.append(public_cards[1])
        p.append(h1)
        p.append(h2)
        rank = rank_eval(p)
        # strength = 

# strength,
    return is_suit, is_pocket, is_high, is_close, rank


def compute_pot(pot, line, currency):
    strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
    pot += float(strings[-1])

    return pot

def get_money(line, name_dic, currency):
    isfold = False
    for n in name_dic.keys():
        if n in line:
            if 'call' in line:
                opt = 'call'
            elif 'bet' in line:
                pattern = r':(.*)'
                match = re.search(pattern, line)
                opt = match.group(1)
            elif 'raise' in line:
                pattern = r':(.*)'
                match = re.search(pattern, line)
                opt = match.group(1)
            elif 'check' in line:
                opt = 'check'
            elif 'fold' in line:
                opt = 'fold'
                isfold = True

            # pattern = r'\{}[\d\.]+'.format(currency)
            pattern = r'\{}\d+\.\d+'.format(currency)
            matches = re.findall(pattern, line)
            if not matches:
                matches = re.findall(r'\{}\d+'.format(currency), line)
            # print(matches)
            
            if len(matches) == 1:
                use_money = float(matches[0][1:])
            elif len(matches) == 2:
                use_money = float(matches[1][1:]) - float(matches[0][1:])
            else:
                use_money = 0
            current_money = name_dic[n][1] - use_money
            name_dic[n][1] = current_money
            return round(current_money,3), opt, isfold
        
def get_info_dic(info_dic, line, name_dic, hand, currency, pot, phase_info, opt, counter, order, phase, public_cards):
    tmp_info_dic = info_dic.copy()
    tmp_info_dic["id"] = counter
    tmp_info_dic["type"] += type_determine(line.split()[1])

    tmp_info_dic["from_human"] += "Order: " + str(order) + "\n"
    if phase == 'PREFLOP':
        tmp_info_dic["from_human"] += "Public cards: [\'**\' \'**\' \'**\' \'**\' \'**\']\n"
    elif phase == 'FLOP':
        tmp_info_dic["from_human"] += "Public cards: ["
        for m in public_cards[0].split():
            tmp_info_dic["from_human"] += "\'" + m + "\' "
        tmp_info_dic["from_human"] += "\'**\' \'**\']\n"
    elif phase == 'TURN':
        tmp_info_dic["from_human"] += "Public cards: ["
        for m in public_cards[0].split():
            tmp_info_dic["from_human"] += "\'" + m + "\' "
        tmp_info_dic["from_human"] += "\'" + public_cards[1] + "\' "
        tmp_info_dic["from_human"] += "\'**\']\n"
    else:
        tmp_info_dic["from_human"] += "Public cards: ["
        for m in public_cards[0].split():
            tmp_info_dic["from_human"] += "\'" + m + "\' "
        tmp_info_dic["from_human"] += "\'" + public_cards[1] + "\']\n"

    is_suit, is_pocket, is_high, is_close, rank = strength_estimation(hand, phase, public_cards)

    for o in order:
        k = "Seat " + o
        if k != "Seat " + get_seat(line, name_dic):
            tmp_info_dic["from_human"] += k + " " + \
                                    "Hand: [\'**\', \'**\']\n" + \
                                    "Money: " + str(phase_info[k][0]) + "\n" + \
                                    "Action: " + str(phase_info[k][1]) + "\n" + \
                                    "Discard: " + str(phase_info[k][2]) + "\n\n"
        else:
            tmp_info_dic["from_human"] += "My seat: [Seat " + get_seat(line, name_dic) + "]\n" + \
                            "My hand: [" + "\'" + hand[:2] + "\', \'" + hand[3:] + "\'" + "]\n"
            
            hand_eval_list = []

            if is_suit:
                hand_eval_list.append('\'suit\'')
            if is_pocket:
                hand_eval_list.append('\'pocket\'')
            if is_high:
                hand_eval_list.append('\'high\'')
            if is_close:
                hand_eval_list.append('\'close\'')
            
            if not is_suit and not is_pocket and not is_high and not is_close:
                tmp_info_dic["from_human"] += "My hand is [\'bad\']\n"
            else:
                tmp_info_dic["from_human"] += "My hand is " + str(hand_eval_list) + "\n"
            tmp_info_dic["from_human"] += "Rank: [\'"+ rank+ "\']\n"
                                        
            tmp_info_dic["from_human"] += "My money: " + str(phase_info[k][0]) + "\n" + \
                            "My Action: " + str(phase_info[k][1]) + "\n" + \
                            "The pot value is [" + currency + str(pot) + ']\n' + \
                            "The choices contain: [\'fold\', \'check\', \'bet\', \'raise\', \'call\'], what should I do?\n" +\
                            "If I choose to \'bet\' or \'raise\', then how much? Give a positive floating point number between (0," + \
                            str(phase_info[k][0][0]) + "]\n\n"
    tmp_info_dic["from_gpt"] += opt

    counter += 1
    return tmp_info_dic, counter

# 对亮牌玩家的一次决策的完整结算流程
def settle(player_info, line, name_dic, currency, info_dic, phase_info, pot, counter, order, phase, public_cards):
    hand = player_info[name_dic[line.rsplit(":",1)[0]][0]][1:-1]
    tmp_seat = "Seat " + get_seat(line, name_dic)
    current_money, opt, isfold = get_money(line, name_dic, currency)
    # 先结算phase_info，再更新phase_info
    tmp_info_dic, counter = get_info_dic(info_dic, line, name_dic, hand, currency, round(pot,3), phase_info, opt, counter, order, phase, public_cards)

    _ = phase_info[tmp_seat][1].append(opt)
    phase_info[tmp_seat][0] = [current_money]
    phase_info[tmp_seat][2] = [isfold]

    strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
    if strings:
        if not any(strings[-1] in name for name in name_dic.keys()):
            pot += float(strings[-1])

    return tmp_info_dic, phase_info, pot, counter

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
# with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_{}/prompt.json'.format(style), 'w') as json_file:
with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_good/prompt.json', 'w') as json_file1, \
    open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad1/prompt.json', 'w') as json_file2, \
    open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad2/prompt.json', 'w') as json_file3, \
    open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad3/prompt.json', 'w') as json_file4, \
    open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad4/prompt.json', 'w') as json_file5, \
    open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad5/prompt.json', 'w') as json_file6: \

# with open('prompt_good.json', 'w') as json_file1, \
    # open('prompt_bad.json', 'w') as json_file2:

    def json_update(name_dic, line):
        for k, v in name_dic.items():
            if v[0] == name_dic[line.rsplit(":",1)[0]][0]:
                if k in top_k:
                    json_file1.write(json_str+',')
                elif k in bottom_ks[0]:
                    json_file2.write(json_str+',')
                    json_file3.write(json_str+',')
                    json_file4.write(json_str+',')
                    json_file5.write(json_str+',')
                    json_file6.write(json_str+',')
                elif k in bottom_ks[1]:
                    json_file3.write(json_str+',')
                    json_file4.write(json_str+',')
                    json_file5.write(json_str+',')
                    json_file6.write(json_str+',')
                elif k in bottom_ks[2]:
                    json_file4.write(json_str+',')
                    json_file5.write(json_str+',')
                    json_file6.write(json_str+',')
                elif k in bottom_ks[3]:
                    json_file5.write(json_str+',')
                    json_file6.write(json_str+',')
                elif k in bottom_ks[4]:
                    json_file6.write(json_str+',')


    json_file1.write('[')
    json_file2.write('[')
    json_file3.write('[')
    json_file4.write('[')
    json_file5.write('[')
    json_file6.write('[')

    counter = 0
    # 对每一局进行处理，若该局中的玩家不包含高手，则一律存入低手数据
    for fid, field in enumerate(fields):
        if fid % 5000 == 0:
            print('field no: ', fid)
        info_dic, name_dic = info_dic_init()
        lines = field.split("\n")
        loser_info = {}
        winner_info = {}
        # 池中注
        pot = 0.0
        
        # 对每一行，暂时跳过showdown和summary，直接从输家赢家附近开始
        for i, line in enumerate(lines):
            if duplicate_process(line):
                continue

            if "PokerStars Hand" in line:
                # info_dic["id"] += re.findall("#.*?:", line)[0][1:-1]

                if "$" in line:
                    currency = "$"
                    pattern = r'\$([\d\.]+)/\$([\d\.]+)'
                    match = re.search(pattern, line)
                    blind_value = match.group(0)
                elif "£" in line:
                    currency = "£"
                    pattern = r'\£([\d\.]+)/\£([\d\.]+)'
                    match = re.search(pattern, line)
                    blind_value = match.group(0)
                elif "€" in line:
                    currency = "€"
                    pattern = r'\€([\d\.]+)/\€([\d\.]+)'
                    match = re.search(pattern, line)
                    blind_value = match.group(0)

                continue

            # 可能是n赢n输的局面，而输家也许不会展示牌面，因此输家只记录有牌面的，赢家都记录，均用字典存放
            # 记录输家位置和手牌
            if "lost" in line.split():
                loser_seat = line.split()[1][0]
                pattern = r'\[\w{2} \w{2}\]'
                match = re.search(pattern, line)
                lose_hand = match.group()
                loser_info[loser_seat] = lose_hand
                continue
            # 如果输了但是盖牌了，则没有输家数据
            elif "mucked" in line.split():
                continue
            # 记录赢家位置和手牌
            elif "won" in line.split():
                winner_seat = line.split()[1][0]
                pattern = r'\[\w{2} \w{2}\]'
                match = re.search(pattern, line)
                win_hand = match.group(0)
                winner_info[winner_seat] = win_hand
                continue
            
            # 当summary信息都遍历了之后，跳出循环，进入下一组循环
            elif "button" in line and "Table" in line and "#" in line:
                break_i = i
                lines = lines[break_i:]
                break

        player_count = 0
        for i, line in enumerate(lines):

            if duplicate_process(line):
                continue

            # 记录座位和人名的关系，在后续对局中替换掉
            if "Seat" in line.split() and "chips)" in line.split():
                # key为人名，value为[位置，钱]
                # print(line)
                match = re.search(r":.*\(", line)
                pattern = r'\{}\d+\.\d+'.format(currency)
                match2 = re.search(pattern, line)
                if not match2:
                    match2 = re.search(r'\{}\d+'.format(currency), line)
                name_dic[match.group()[2:-2]] = [line.split()[1][0], round(float(match2.group()[1:]),3)]
                player_count += 1
                continue

            if "posts small" in line:
                break_i = i
                lines = lines[break_i:]
                break


######## 录入之前的信息 #########
        info_dic["from_human"] += "Player amount: [" + str(player_count) + "]\n"
        info_dic["from_human"] += "Blind value: [" + blind_value + "]\n"

        phase_info = phase_info_init(name_dic)

######## 每个阶段的对局信息 ##########
        for i, line in enumerate(lines):
            isfold = False
            if duplicate_process(line):
                continue

            #######################################################################
            # PREFLOP ####### 这个阶段主要是算一算pot
            if "posts small blind" in line:
                order = get_order(line, name_dic)
                pot = compute_pot(pot, line, currency)
                phase = 'PREFLOP'
                public_cards = ''
                continue
            elif "posts big blind" in line:
                pot = compute_pot(pot, line, currency)
                continue
            elif "posts small & big blind" in line:
                order = get_order(line, name_dic)
                pot = compute_pot(pot, line, currency)
                phase = 'PREFLOP'
                public_cards = ''

                continue


        ##### PREFLOP #####
            if "* FLOP *" in line or "FIRST FLOP" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif "* HOLE CARDS *" in line:
                continue
            # 这里是找到位置信息
            elif name_dic[line.rsplit(":",1)[0]][0] in loser_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(loser_info, line, name_dic,
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                
                # 分数据集
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue

            elif name_dic[line.rsplit(":",1)[0]][0] in winner_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(winner_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
            else:
                tmp_seat = "Seat " + get_seat(line, name_dic)
                current_money, opt, isfold = get_money(line, name_dic, currency)

                _ = phase_info[tmp_seat][1].append(opt)
                phase_info[tmp_seat][0] = [current_money]
                phase_info[tmp_seat][2] = [isfold]

                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue


        # 在下一阶段开始前，去除fold的玩家
        for k in phase_info.keys():
            if phase_info[k][2] or k[-1] not in order:
                continue
            else:
                order.remove(k[-1])

        ###### FLOP ######
        for i, line in enumerate(lines):
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成flop阶段的行为prompt，然后分别制作赢家和输家的prompt

            if "* FIRST FLOP *" in line:
                break
            elif "* FLOP *" in line:
                phase = 'FLOP'
                pattern = r'\[([^\]]+)\]'
                public_cards = re.findall(pattern, line)
                continue

            if "* TURN *" in line or "* FIRST TURN *" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif name_dic[line.rsplit(":",1)[0]][0] in loser_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(loser_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter, 
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            elif name_dic[line.rsplit(":",1)[0]][0] in winner_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(winner_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            else:
                tmp_seat = "Seat " + get_seat(line, name_dic)
                current_money, opt, isfold = get_money(line, name_dic, currency)

                _ = phase_info[tmp_seat][1].append(opt)
                phase_info[tmp_seat][0] = [current_money]
                phase_info[tmp_seat][2] = [isfold]

                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue



        # 在下一阶段开始前，去除fold的玩家
        for k in phase_info.keys():
            if phase_info[k][2] or k[-1] not in order:
                continue
            else:
                order.remove(k[-1])

        for i, line in enumerate(lines):
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成turn阶段的行为prompt，然后分别制作赢家和输家的prompt
            
            if "* FIRST TURN *" in line or "FIRST FLOP" in line:
                break
            elif "* TURN *" in line:
                pattern = r'\[([^\]]+)\]'
                public_cards = re.findall(pattern, line)
                phase = 'TURN'
                
                continue

            if "* RIVER *" in line or "* FIRST RIVER *" in line:
                break_i = i
                lines = lines[break_i:]
                break
            elif name_dic[line.rsplit(":",1)[0]][0] in loser_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(loser_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            elif name_dic[line.rsplit(":",1)[0]][0] in winner_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(winner_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            else:
                tmp_seat = "Seat " + get_seat(line, name_dic)
                current_money, opt, isfold = get_money(line, name_dic, currency)

                _ = phase_info[tmp_seat][1].append(opt)
                phase_info[tmp_seat][0] = [current_money]
                phase_info[tmp_seat][2] = [isfold]

                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue




        # 在下一阶段开始前，去除fold的玩家
        for k in phase_info.keys():
            if phase_info[k][2] or k[-1] not in order:
                continue
            else:
                order.remove(k[-1])
        # phase_info = {}

        for i, line in enumerate(lines):
            if duplicate_process(line):
                continue

            #######################################################################
            # 生成RIVER阶段的行为prompt，然后分别制作赢家和输家的prompt
            # 如果出现FIRST RIVER字眼，说明之前已经都all in了，之后不需要再制作对应的prompt
            if "* FIRST RIVER *" in line or "FIRST TURN" in line or "FIRST FLOP" in line:
                break
            elif "* RIVER *" in line:
                phase = 'RIVER'
                pattern = r'\[([^\]]+)\]'
                public_cards = re.findall(pattern, line)

                continue

            elif name_dic[line.rsplit(":",1)[0]][0] in loser_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(loser_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            elif name_dic[line.rsplit(":",1)[0]][0] in winner_info.keys():
                tmp_info_dic, phase_info, pot, counter = settle(winner_info, line, name_dic, 
                                                         currency, info_dic, phase_info, pot, counter,
                                                         order, phase, public_cards)
                json_str = json.dumps(tmp_info_dic, indent=1)
                json_update(name_dic, line)

                # 结算后重新整理文本
                tmp_info_dic = {}
                continue
            else:
                tmp_seat = "Seat " + get_seat(line, name_dic)
                current_money, opt, isfold = get_money(line, name_dic, currency)

                _ = phase_info[tmp_seat][1].append(opt)
                phase_info[tmp_seat][0] = [current_money]
                phase_info[tmp_seat][2] = [isfold]

                strings = [match.group(1) for match in re.finditer(r"\{}([\w.]+)".format(currency), line)]
                if strings:
                    if not any(strings[-1] in name for name in name_dic.keys()):
                        pot += float(strings[-1])
                continue
        if fid == len(fields)-3:
        # if fid == test_round:
            break

    json_file1.write(']')
    json_file2.write(']')
    json_file3.write(']')
    json_file4.write(']')
    json_file5.write(']')
    json_file6.write(']')

# with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_{}/prompt.json'.format(style), 'r+') as f:
# with open('prompt_good.json', 'r+') as f1:
with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_good/prompt.json', 'r+') as f1:
    content = f1.read()
    content = content[:-2]+']'
    f1.seek(0)
    f1.truncate()
    f1.write(content)

# with open('prompt_bad.json', 'r+') as f2:
for i in range(1,6):
    with open('/root/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/Local/TH_bad{}/prompt.json'.format(i), 'r+') as f:
        content = f.read()
        content = content[:-2]+']'
        f.seek(0)
        f.truncate()
        f.write(content)

print('process time: ', time.time()-start_time)


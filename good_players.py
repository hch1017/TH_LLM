import re
import os
import matplotlib.pyplot as plt

path = './BulkHands'
files = os.listdir(path)

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
                # print(line.split())
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

print(len(player_dic.copy().keys()))

# 删掉局数少的玩家
round_count = 100
ks = player_dic.copy().keys()
for k in ks:
    if player_dic.get(k)[1] < round_count:
        del player_dic[k]


top10 = sorted(player_dic.items(), key=lambda x: x[1][0]/x[1][1], reverse=True)[:10]
bottom = sorted(player_dic.items(), key=lambda x: x[1][0]/x[1][1], reverse=False)[:200]

big10 = sorted(player_dic.items(), key=lambda x: x[1][0], reverse=True)[:10]

print(len(player_dic.copy().keys()))

# print('big10:', big10)
# print('top10:', top10)
# print('bottom200:', bottom)


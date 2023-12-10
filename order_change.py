import os

path = './BulkHands'
files = os.listdir(path)
total_f = open("showdown.txt", "w")

########  遍历所有文件，将有SHOW DOWN的对局中从SHOW DOWN开始的信息调换到对局开始处，然后存到一个文件中 #######

for f_name in files:
    f = open(path+'/'+f_name)
    content = f.read()
    f.close()

    fields = content.split('\n\n')
    
    # 对当前文件中每一个对局进行处理
    for field in fields:
        lines = field.split("\n")
        flag = 0
        # 查看对局的每一行
        for i, line in enumerate(lines):
            if 'SHOW DOWN' in line:
                # print('Showdown')
                flag = 1
                break
        
        # 有SHOW DOWN的对局换序后存到一个文件中
        if flag == 1:
            line_to_move = lines[i:]
            del lines[i:]
            if 'PokerStars Hand' in lines[1]:
                lines = [lines[1]] + line_to_move + lines[2:]
            elif 'PokerStars Hand' in lines[0]:
                lines = [lines[0]] + line_to_move + lines[1:]
            if 'PokerStars Hand' in lines[2]:
                print(i)
            new_text = "\n".join(lines) + '\n\n\n'
            total_f.write(new_text)
        else:
            continue
total_f.close()

import re
f = open('../data/training_text')

for doc in f:
 # print(doc[:20])
 m = re.match(r'^\d+\|\|(.*)$',doc)
 if m: 
    content = m.group(1) 
    # print(content[:20])
    sentences = content.split('.')
    for s in sentences:
        # expression pattern 1       
        exp_behavioral = r"([^\s\,]+\s)(behav|role|function|activity|operat|working|capacity)([^\s\,]*)"
        # exp_behavioral = r"(behav|role|function|activity|operat|working|capacity)([^\s\,]*)"
        exp_change = r"(chang|turn|render|alter|vary|differ|switch)([^\s\,]*)(\s[^\s\,]+)"
        # exp_change = r"(chang|turn|render|alter|vary|differ|switch)([^\s\,]*)"
        exp_gap = r"(\s|\s[^\s]+\s|\s[^\s]+\s[^\s]+\s)"
        # exp_merge = exp_behavioral + exp_gap + exp_change

        exp_merge = exp_behavioral + exp_gap + exp_change
        exps = re.findall(exp_merge, s)
        for exp in exps:
            print(exp[0], end=' ')
            trio = [exp[1]+exp[2], exp[3], exp[4]+exp[5]]
            print(trio, end='') 
            print(' ' + exp[6])  


        # expression pattern 2
        exp_behavioral = r"(behav|role|function|activity|operat|working|capacity)([^\s\,]*)(\s[^\s\,]+)"
        # exp_behavioral = r"(behav|role|function|activity|operat|working|capacity)([^\s\,]*)"
        exp_change = r"([^\s\,]+\s)(chang|turn|render|alter|vary|differ|switch)([^\s\,]*)"
        # exp_change = r"(chang|turn|render|alter|vary|differ|switch)([^\s\,]*)"
        exp_gap = r"(\s|\s[^\s]+\s|\s[^\s]+\s[^\s]+\s)"
        # exp_merge = exp_behavioral + exp_gap + exp_change

        exp_merge = exp_change + exp_gap + exp_behavioral
        exps = re.findall(exp_merge, s)
        for exp in exps:
            print(exp[0], end=' ')
            trio = [exp[1]+exp[2], exp[3], exp[4]+exp[5]]
            print(trio, end='') 
            print(' ' + exp[6])         
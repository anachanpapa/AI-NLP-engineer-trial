import re
f = open('../data/training_text')

for doc in f:
 # print(doc[:20])
 m = re.match(r'^\d+\|\|(.*)$',doc)
 if m: 
    content = m.group(1) 
    sentences = content.split('.')
    for s in sentences:
        if re.search(r'behav|role|function|activity|operat|working|capacity|chang|turn|render|alter|vary|differ|switch', s):
            print(s.strip())        
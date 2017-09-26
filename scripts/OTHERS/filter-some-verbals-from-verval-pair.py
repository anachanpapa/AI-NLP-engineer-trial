import re

f = open('../data/processed_data/verbal-pair.uniq', 'r')
ignore = open('../data/processed_data/verbal-pair.uniq.vocab.ignore', 'r')

ignore_list = []
for word in ignore:
    word = word.replace('\n','') 
    word = word.strip() 
    ignore_list.append(word)

ignore_select = '|'.join(ignore_list)
for line in f:
    line = line.replace('\n','') 
    line = line.strip()
    if not re.search(ignore_select, line):
        print(line)

f.close()
ignore.close()

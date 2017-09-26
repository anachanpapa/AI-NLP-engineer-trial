import re
# target = open('./dummy.pair')
target = open('../data/processed_data/training_text.patten.m5s0S2')
verbal = open('../data/processed_data/training_text.verbal.clean')

verbals = []
for line in verbal:
    line = line.replace('\n','') 
    line = line.strip()
    if re.match("^is|are|was|were|be|been|being|do|doing|did|done|does|have|has|having|had$", line):
        continue
    verbals.append(line)

select_verbals = '(^|\s+)(' + '|'.join(verbals) + ')(\s+|$)'
# select_verbals = '(' + '|'.join(verbals) + ')'
verbal_pair = select_verbals + '.+' + select_verbals
verbal_pair2 = '(^|\s+)(' + '|'.join(verbals) + ')(\s+)(' + '|'.join(verbals) + ')(\s+|$)'


for snippet in target:
    if not re.match("^\s?[a-z]+|[A-Z]+", snippet): continue
    snippet = snippet.replace('\n','') 
    snippet = snippet.strip()
    snippet_no_count = re.sub(r'/[0-9]+', '', snippet)
    # print('kuur')
    # if re.search(verbal_pair, snippet):
    m = re.findall(verbal_pair, snippet_no_count)
    if len(m) > 0:
        # print(snippet, m)     
        print(m[0][1], m[0][4])   
        continue

    m = re.findall(verbal_pair2, snippet_no_count)
    if len(m) > 0:
        # print(snippet, m) 
        print(m[0][1], m[0][3])             
        continue

target.close()            
verbal.close()            
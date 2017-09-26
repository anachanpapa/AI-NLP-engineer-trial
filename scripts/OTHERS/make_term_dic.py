import re
from nltk.corpus import stopwords
from nltk import stem

f = open('../data/processed_data/term.synonym', 'r')
d = open('../data/processed_data/term.dic', 'w')

dic = {}
stemmer = stem.PorterStemmer()
for line in f:
    line = line.replace('\n','') 
    line = line.strip() 
    line = line.replace('|', ' ')
    line = line.replace(':', ' ')
    line = line.replace('/', ' ')
    line = re.sub(r',|\.|;|<|>|\(|\)|\#|%|\$|\"|\'|\*', '', line) 
    line = re.sub(r'\s*-\d+-\s*', '', line) 
    line = line.lower()
    items = line.split(' ')
    for item in items: 
        if item in stopwords.words('english'): continue
        if len(item) < 3: continue
        dic[item] = 1

stemmed = {}
for item in dic:
    stemmed[stemmer.stem(item)] = 1


for item in sorted(stemmed.keys()):
    d.write(item + '\n')

f.close()
d.close()
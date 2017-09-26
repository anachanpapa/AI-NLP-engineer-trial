import nltk
import subprocess
import re
import sys
from nltk import stem

# fall = open('../data/processed_data/test_text.vocab.alphabet', 'r')
# fverbal = open('../data/processed_data/test_text.verbal', 'w')
# felse = open('../data/processed_data/test_text.else', 'w')

# fall = open('../data/processed_data/training_text.vocab', 'r')
# fverbal = open('../data/processed_data/training_text.verbal', 'w')
# felse = open('../data/processed_data/training_text.else', 'w')

fall = open('../data/processed_data/both_phrases.vocab', 'r')
fverbal = open('../data/processed_data/both_phrases.verbal', 'w')
felse = open('../data/processed_data/both_phrases.else', 'w')

def lookupWordNet(exp):
    args = ['wordnet', exp, '-derin']
    res = subprocess.run(args, stdout=subprocess.PIPE)
    # print(res.stdout)
    # print(type(res.stdout.decode('utf-8')))
    # if re.search(r'verb', res.stdout.decode('utf-8')):
    for m in re.findall(r'\(verb\)\s(\w+)', res.stdout.decode('utf-8')):
        # print(derive_from, exp)
        stemmer = stem.PorterStemmer()
        derive_from = stemmer.stem(m)
        if derive_from in exp: return 1

    else:
        args = ['wordnet', exp, '-deriv']
        res = subprocess.run(args, stdout=subprocess.PIPE)
        # if re.search(r'verb', res.stdout.decode('utf-8')):
        if len(res.stdout.decode('utf-8')) > 0:
            return 1
        else:
            return 0    

clean = []
for word in fall:
    if re.match(r'^\w+$', word): clean.append(word.lower())
clean = list(set(clean))
print(len(clean))


for i in range(len(clean)):
    word = clean[i]
    print('--- ' + str(i) + ' ---')
    text = nltk.word_tokenize(word)
    tag = nltk.pos_tag(text)
    exp = tag[0][0]
    pos = tag[0][1]
    if 'VB' in pos: 
        fverbal.write(exp + '\n')
    elif lookupWordNet(exp):
        fverbal.write(exp + '\n') 
        # print(exp)   
    else:   
        felse.write(exp + '\n')   

fall.close()    
fverbal.close()    
felse.close()    


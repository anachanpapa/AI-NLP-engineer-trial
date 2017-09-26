import re
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import stem
stemmer = stem.LancasterStemmer()
#-----------  MINING PHRASES FROM TRAINING DATA --------------


def getSynonyms(word):
    syns = wn.synsets(word)
    synonyms = [item.lemmas()[0].name() for item in syns]
    return list(set(synonyms))
    # print(unknown + ': ')
    # print(list(set(synonyms)))

def topFreq(wordList):
    freqs = {}
    for word in wordList:
        freqs[word] = train_dic[word]
    for k in reversed(sorted(freqs, key=lambda k:freqs[k])):
        if freqs[k] > 0: return k
    return None

train_phrases = open('../data/training_text', 'r')
test_phrases = open('../data/test_text', 'r')
train_phrases.readline()
test_phrases.readline()

train_dic = defaultdict(int)
test_dic = defaultdict(int)

for line in train_phrases:
    line = line.replace('\n','') 
    line = line.strip()
    for word in line.split(' '):
        train_dic[word] += 1

for line in test_phrases:
    line = line.replace('\n','') 
    line = line.strip()
    for word in line.split(' '):
        test_dic[word] += 1        

train_phrases.close()
test_phrases.close()

train_dic_filtered = defaultdict(int)
for word in train_dic.keys():
    if train_dic[word] >= 1: train_dic_filtered[word] = train_dic[word]

test_dic_filtered = defaultdict(int)
for word in test_dic.keys():
    if test_dic[word] >= 1: test_dic_filtered[word] = test_dic[word]
 
exist_in_both = defaultdict(int)
not_exist_in_train = defaultdict(int)
for word in sorted(test_dic_filtered.keys()): 
    if word in train_dic_filtered: 
        exist_in_both[word] = 1
    else:
        not_exist_in_train[word] = 1

print('not_exist_in_train / test_dic: ', end='')
print(len(not_exist_in_train)/len(test_dic_filtered), end='-->')
print(str(len(not_exist_in_train)) + '/' + str(len(test_dic_filtered)))

train_phrases.close()
test_phrases.close()


translate = {}

def findSynonym(unknown):

    synonyms = getSynonyms(unknown)
    if len(synonyms) > 0:
        top = topFreq(synonyms)
        if top:
            return top
        else:
            for word in synonyms:
                synonyms = getSynonyms(word)
                if len(synonyms) > 0:
                    top = topFreq(synonyms)
                    if top:
                        return top
                    else:
                        for word in synonyms: 
                            synonyms = getSynonyms(word)
                            if len(synonyms) > 0:
                                return topFreq(synonyms)
    else:
        return None            

for unknown in not_exist_in_train:
    try:
        found = findSynonym(unknown)
        if found != None:
            translate[unknown] = found
            # print(unknown, translate[unknown] )        
        else:
            stemmed = stemmer.stem(unknown)
            if stemmed in train_dic_filtered:
                translate[unknown] = stemmed
            else:
                 findSynonym(stemmed)   
    except:
        pass

print('Number of word swapping found: ' + str(len(translate)))

for k in translate.keys():
    print(k, translate[k])      
test_phrases = open('../data/test_text', 'r')
test_phrases_translated = open('../data/test_text.translated', 'w')
test_phrases.readline()
test_phrases_translated.write('ID,Text \n')
i = 0
for l in test_phrases:
    docid = ''
    line = ''
    m = re.match(r'^(\d+)\|\|(.*)$',l)
    if m: 
        docid = m.group(1)  
        line = m.group(2) 

    line = line.replace('\n','') 
    line = line.strip()
    words = []
    for word in line.split(' '):
        if word in exist_in_both:
            words.append(word)
        elif word in not_exist_in_train:
            if word in translate:
                words.append(translate[word])
            else:
                words.append(word)               
    translated = docid + '||' +  ' '.join(words)  
    test_phrases_translated.write(translated + '\n')
    i += 1

test_phrases.close()
test_phrases_translated.close()




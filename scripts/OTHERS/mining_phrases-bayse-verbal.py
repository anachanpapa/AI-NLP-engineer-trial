import re
from collections import defaultdict
from math import log 

# target = open('../data/test_text')
# target = open('../data/training_text')

#-----------  MINING PHRASES FROM TRAINING DATA --------------

target = open('../data/training_text', 'r')
gene_variation = open('../data/training_variants', 'r')
verbal = open('../data/processed_data/both_phrases.verbal')
ignore = open('../data/processed_data/verbal-pair.uniq.vocab.ignore', 'r')

geneMap = {}
variationMap = {}
for line in gene_variation:
    line = line.replace('\n','') 
    line = line.strip()
    [docid, gene, variation, mclass] = line.split(',')
    geneMap[docid] = gene
    variationMap[docid] = variation

ignores =[]
for line in ignore:
    line = line.replace('\n','') 
    line = line.strip()
    ignores.append(line)


verbals = []
for line in verbal:
    line = line.replace('\n','') 
    line = line.strip()
    # if re.match("^" + "|".join(ignores) + "$", line):
    #     continue
    verbals.append(line)

select_verbals = '(\s+)(' + '|'.join(verbals) + ')(\s+)'
# select_verbals = '(' + '|'.join(verbals) + ')'
verbal_pair = select_verbals + '(.{1,30})' + select_verbals
verbal_pair2 = '(\s+)(' + '|'.join(verbals) + ')(\s+)(' + '|'.join(verbals) + ')(\s+)'

word_count_variation = defaultdict(int) 
word_prob_variation = defaultdict(float) 
totall_count_variation = 0
word_count_no_variation = defaultdict(int) 
word_prob_no_variation = defaultdict(float) 
totall_count_no_variation = 0

target.readline()
for doc in target:
    doc = doc.replace('\n','') 
    doc = doc.strip()
    m = re.match(r'^(\d+)\|\|(.*)$',doc)
    if m: 
        docid = m.group(1)  
        content = m.group(2) 
        sentences = content.split('.')
        phrases = []
        for s in sentences:
            s = re.sub(r'\W+', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            s = s.replace(',', '')
            s = s.lower()
            if re.search(variationMap[docid], s):
                words = s.split(' ')
                for word in words:
                    totall_count_variation += 1
                    word_count_variation[word] += 1

            else:
                words = s.split(' ')
                for word in words:
                    totall_count_no_variation += 1
                    word_count_no_variation[word] += 1


for word in word_count_variation:
    word_prob_variation[word] = word_count_variation[word]/totall_count_variation

for word in word_count_no_variation:
    word_prob_no_variation[word] = word_count_no_variation[word]/totall_count_no_variation


def filter_umimportant(wordList):
    scoreMap = {}
    filtered = []
    for word in wordList:
        if word_prob_no_variation[word] == 0:
            filtered.append(word)
        else:    
            scoreMap[word] = word_prob_variation[word]/word_prob_no_variation[word]
    i = 0
    for k in reversed(sorted(scoreMap, key=lambda k:scoreMap[k])): 
        filtered.append(k)
        i += 1
        if i > int(len(wordList) * 0.6): break 

    return filtered    


def scoring(wordList, docid):
    VariationProb = log(1)
    NotVariationProb = log(1)
    for word in wordList:
        if word in word_count_variation.keys() and word in word_count_no_variation.keys():
            VariationProb += log(word_prob_variation[word])
            NotVariationProb += log(word_prob_no_variation[word])
    score = VariationProb - NotVariationProb

    phrase = ' '.join(wordList)
    phrase = phrase.lower()
    targetVariation = variationMap[docid]
    targetVariation = targetVariation.lower()
    target_prefix = targetVariation[:4]
    if re.search(targetVariation, phrase):
        score += 50000
        return score

    if re.search(r'\s+|_', targetVariation):
        parts = re.split(r'\s+|_', targetVariation)   
        for part in parts:
            if re.search(part, phrase):
                score += 40000
                return score

    if re.search(target_prefix, phrase):
        score += 30000
        return score

    m = re.match(r'^\D*(\d+)\D*$', targetVariation)
    if m:
        num_sequence = '999999'
        if m:
            num_sequence = m.group(1)
            if re.search(num_sequence, phrase): 
                score += 20000
                return score
        else:
            targetGene = geneMap[docid]
            targetGene = targetGene.lower()
            if re.search(targetGene, phrase): 
                score += 10000
                return score

    return score

target.close()
gene_variation.close()

target = open('../data/training_text', 'r')
gene_variation = open('../data/training_variants', 'r')
training_phrases = open('../data/processed_data/training_phrases', 'w')

training_phrases.write('ID,Text\n')
target.readline()
for doc in target:
    doc = doc.replace('\n','') 
    doc = doc.strip()
    m = re.match(r'^(\d+)\|\|(.*)$',doc)
    if m: 
        docid = m.group(1)  
        content = m.group(2) 
        sentences = content.split('.')
        scores = {}
        print('------------------  '  + docid + '  ---------------------')
        for s in sentences:
            # s = s.replace(',', '')
            s = re.sub(r',|\(|\)', '', s)
            s = s.lower()
            words = s.split(' ')
            scores[s] = scoring(words, docid)
            
        i = 0
        top5senences = [] 
        for k in reversed(sorted(scores, key=lambda k:scores[k])): 
            # print(scores[k], k)
            wordList = k.split(' ')
            # wordList = filter_umimportant(wordList)
            sentence_plus = wordList
            # print(sentence_plus)
            # top5senences = top5senences + sentence_plus

            for word in sentence_plus:
                if word in verbals: top5senences.append(word)

            # hits = re.findall(select_verbals, ' '.join(sentence_plus))
            # if len(hits) > 0:
            # if re.search(geneMap[docid], s) or re.search(variationMap[docid], s):
                # phrases.append(sentence_plus)
                # print(hits) 
                # continue

            # hits = re.findall(verbal_pair2, ' '.join(sentence_plus))
            # if len(hits) > 0:
            #     # if re.search(geneMap[docid], s) or re.search(variationMap[docid], s):    
            #     # phrases.append(sentence_plus)
            #     print(hits)   
            #     continue

            # if re.search(variationMap[docid], k): 
                # sentence_plus = sentence_plus + [word + '_Variation' for word in wordList if not word in ignores]
                # if re.search(geneMap[docid], k):
                #     sentence_plus = sentence_plus + [word + '_Gene' for word in wordList if not word in ignores]
            i += 1
            if i >= 3: break
        training_phrases.write(docid + '||' + ' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
training_phrases.close()

notexist()

#-----------  MINING PHRASES FROM TEST DATA --------------

target = open('../data/test_text', 'r')
gene_variation = open('../data/test_variants', 'r')
verbal = open('../data/processed_data/both_phrases.verbal')
ignore = open('../data/processed_data/verbal-pair.uniq.vocab.ignore', 'r')

geneMap = {}
variationMap = {}
for line in gene_variation:
    line = line.replace('\n','') 
    line = line.strip()
    [docid, gene, variation] = line.split(',')
    geneMap[docid] = gene
    variationMap[docid] = variation

ignores =[]
for line in ignore:
    line = line.replace('\n','') 
    line = line.strip()
    ignores.append(line)


verbals = []
for line in verbal:
    line = line.replace('\n','') 
    line = line.strip()
    if re.match("^" + "|".join(ignores) + "$", line):
        continue
    verbals.append(line)

select_verbals = '(\s+)(' + '|'.join(verbals) + ')(\s+)'
# select_verbals = '(' + '|'.join(verbals) + ')'
verbal_pair = select_verbals + '(.{1,30})' + select_verbals
verbal_pair2 = '(\s+)(' + '|'.join(verbals) + ')(\s+)(' + '|'.join(verbals) + ')(\s+)'

word_count_variation = defaultdict(int) 
word_prob_variation = defaultdict(float) 
totall_count_variation = 0
word_count_no_variation = defaultdict(int) 
word_prob_no_variation = defaultdict(float) 
totall_count_no_variation = 0

target.readline()
for doc in target:
    doc = doc.replace('\n','') 
    doc = doc.strip()
    m = re.match(r'^(\d+)\|\|(.*)$',doc)
    if m: 
        docid = m.group(1)  
        content = m.group(2) 
        sentences = content.split('.')
        phrases = []
        for s in sentences:
            s = s.replace(',', '')
            s = s.lower()
            if re.search(variationMap[docid], s):
                words = s.split(' ')
                for word in words:
                    totall_count_variation += 1
                    word_count_variation[word] += 1

            else:
                words = s.split(' ')
                for word in words:
                    totall_count_no_variation += 1
                    word_count_no_variation[word] += 1


for word in word_count_variation:
    word_prob_variation[word] = word_count_variation[word]/totall_count_variation

for word in word_count_no_variation:
    word_prob_no_variation[word] = word_count_no_variation[word]/totall_count_no_variation


target.close()
gene_variation.close()

target = open('../data/test_text', 'r')
gene_variation = open('../data/test_variants', 'r')
test_phrases = open('../data/processed_data/test_phrases', 'w')
target.readline()
for doc in target:
    doc = doc.replace('\n','') 
    doc = doc.strip()
    m = re.match(r'^(\d+)\|\|(.*)$',doc)
    if m: 
        docid = m.group(1)  
        content = m.group(2) 
        sentences = content.split('.')
        scores = {}
        print('------------------  '  + docid + '  ---------------------')
        for s in sentences:
            # s = s.replace(',', '')
            s = re.sub(r',|\(|\)', '', s)
            s = s.lower()
            words = s.split(' ')
            scores[s] = scoring(words, docid)
            
        i = 0
        top5senences = [] 
        for k in reversed(sorted(scores, key=lambda k:scores[k])): 
            # print(scores[k], k)
            wordList = k.split(' ')
            # wordList = filter_umimportant(wordList)
            sentence_plus = wordList
            # print(sentence_plus)
            # top5senences = top5senences + sentence_plus

            for word in sentence_plus:
                if word in verbals: top5senences.append(word)

            i += 1
            if i >= 3: break
        test_phrases.write(' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
test_phrases.close()

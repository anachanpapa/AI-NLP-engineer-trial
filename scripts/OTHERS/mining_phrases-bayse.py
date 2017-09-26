import re
from collections import defaultdict
from math import log 


#-----------  MINING PHRASES FROM TRAINING DATA --------------

target = open('../data/training_text', 'r')
gene_variation = open('../data/training_variants', 'r')

geneMap = {}
variationMap = {}
for line in gene_variation:
    line = line.replace('\n','') 
    line = line.strip()
    [docid, gene, variation, mclass] = line.split(',')
    geneMap[docid] = gene
    variationMap[docid] = variation

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
target.readline()
gene_variation = open('../data/training_variants', 'r')
training_phrases = open('../data/processed_data/training_phrases', 'w')
training_phrases.write('ID,Text\n')
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
            wordList = filter_umimportant(wordList)
            sentence_plus = wordList
            top5senences = top5senences + sentence_plus
            i += 1
            if i >= 300: break
        training_phrases.write(docid + '||' + ' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
training_phrases.close()


#-----------  MINING PHRASES FROM TEST DATA --------------

target = open('../data/test_text', 'r')
gene_variation = open('../data/test_variants', 'r')

geneMap = {}
variationMap = {}
for line in gene_variation:
    line = line.replace('\n','') 
    line = line.strip()
    [docid, gene, variation] = line.split(',')
    geneMap[docid] = gene
    variationMap[docid] = variation

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
target.readline()
gene_variation = open('../data/test_variants', 'r')
test_phrases = open('../data/processed_data/test_phrases', 'w')
test_phrases.write('ID,Text\n')
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
            wordList = filter_umimportant(wordList)
            sentence_plus = wordList
            top5senences = top5senences + sentence_plus
            i += 1
            if i >= 300: break
        test_phrases.write(docid + '||' + ' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
test_phrases.close()

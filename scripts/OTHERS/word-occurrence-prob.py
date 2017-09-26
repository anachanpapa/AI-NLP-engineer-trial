import re
from collections import defaultdict
from math import log 

# target = open('../data/test_text')
# target = open('../data/training_text')

#-----------  MINING PHRASES FROM TRAINING DATA --------------

target = open('../data/training_text', 'r')
gene_variation = open('../data/training_variants', 'r')
verbal = open('../data/processed_data/training_text.verbal.clean')
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


def scoring(wordlist):
    VariationProb = log(1)
    NotVariationProb = log(1)
    for word in wordlist:
        if word in word_count_variation.keys() and word in word_count_no_variation.keys():
            VariationProb += log(word_prob_variation[word])
            NotVariationProb += log(word_prob_no_variation[word])
    score = VariationProb - NotVariationProb
    return score

target.close()
gene_variation.close()

target = open('../data/training_text', 'r')
gene_variation = open('../data/training_variants', 'r')
trainig_pharses = open('../data/processed_data/training_phrases', 'w')
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
            words = s.split(' ')
            scores[s] = scoring(words)
            
        i = 0
        top5senences = [] 
        for k in reversed(sorted(scores, key=lambda k:scores[k])): 
            # print(scores[k], k)
            sentence_plus = k

            hits = re.findall(verbal_pair, sentence_plus)
            if len(hits) > 0:
            # if re.search(geneMap[docid], s) or re.search(variationMap[docid], s):
                phrases.append(sentence_plus)
                print(hits) 
                continue

            hits = re.findall(verbal_pair2, sentence_plus)
            if len(hits) > 0:
                # if re.search(geneMap[docid], s) or re.search(variationMap[docid], s):    
                phrases.append(sentence_plus)
                print(hits)   
                continue



            # wordlist = sentence_plus.split(' ')
            # if re.search(geneMap[docid], k):
            #     sentence_plus += ' ' 
            #     sentence_plus += ' '.join([word + '_TargetGene' for word in wordlist if not word in ignores])
            # if re.search(variationMap[docid], k):
            #     sentence_plus += ' ' 
            #     sentence_plus += ' '.join([word + '_TargetVariation' for word in wordlist if not word in ignores])
            # # print(sentence_plus)
            top5senences.append(sentence_plus)

            i += 1
            # print(s)
            if i >= 5: break

        # trainig_pharses.write(docid + '||' + ' '.join(top5senences) + '\n')
        trainig_pharses.write(' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
trainig_pharses.close()

notexist()

#-----------  MINING PHRASES FROM TEST DATA --------------

target = open('../data/test_text', 'r')
gene_variation = open('../data/test_variants', 'r')
ignore = open('../data/processed_data/verbal-pair.uniq.vocab.ignore', 'r')
ignores =[]
for line in ignore:
    line = line.replace('\n','') 
    line = line.strip()
    ignores.append(line)

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
test_pharses = open('../data/processed_data/test_pharses', 'w')
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
            words = s.split(' ')
            scores[s] = scoring(words)
            
        i = 0
        top5senences = [] 
        for k in reversed(sorted(scores, key=lambda k:scores[k])): 
            # print(scores[k], k)
            sentence_plus = k
            wordlist = sentence_plus.split(' ')
            if re.search(geneMap[docid], k):
                sentence_plus += ' ' 
                sentence_plus += ' '.join([word + '_TargetGene' for word in wordlist if not word in ignores])
            if re.search(variationMap[docid], k):
                sentence_plus += ' ' 
                sentence_plus += ' '.join([word + '_TargetVariation' for word in wordlist if not word in ignores])
            # print(sentence_plus)
            top5senences.append(sentence_plus)

            i += 1
            # print(s)
            if i >= 5: break

        # test_pharses.write(docid + '||' + ' '.join(top5senences) + '\n')
        test_pharses.write(' '.join(top5senences) + '\n')

target.close()
gene_variation.close()    
test_pharses.close()
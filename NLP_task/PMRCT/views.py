from django.shortcuts import render
from django.http import HttpResponse
from .models import TrainingData, Behavior, Change, Middle, Regex
import os
import json
import codecs 
import re
import time

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, sent_tokenize, pos_tag

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn. model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from django.views.decorators.csrf import ensure_csrf_cookie

unused_content = []
regex_strs = []
fixed_role = ''
def bootstrapping(request):

    data_initialize()

    make_regex()
    return render(request, 'PMRCT/bootstrapping.html', 
        {
        'unused_content': unused_content,
        'regex_strs': regex_strs,
        'fixed_role': fixed_role,
        })

def data_initialize():
    regexs = Regex.objects.all()
    for item in regexs: item.delete()



def make_regex():
    unused = find_unused_expressions()
    # print(unused)
    role = unused['role']
    global fixed_role
    fixed_role = role
    content = unused['content']
    global unused_content 
    unused_content = content
    regexs = []
    if role == 'behavior':
        used_change = Change.objects.filter(isUsed=1)
        used_change_exps = [exp.expression for exp in used_change]
        used_middle = Middle.objects.filter(isUsed=1)
        used_middle_exps = [exp.expression for exp in used_middle]
        regexs = conbine_expression(role, content, used_middle_exps, used_change_exps)

    elif role == 'change':
        used_behavior = Behavior.objects.filter(isUsed=1)
        used_behavior_exps = [exp.expression for exp in used_behavior]
        used_middle = Middle.objects.filter(isUsed=1)
        used_middle_exps = [exp.expression for exp in used_middle]
        regexs = conbine_expression(role, used_behavior_exps, used_middle_exps, content)

    elif role == 'middle':
        used_change = Change.objects.filter(isUsed=1)
        used_change_exps = [exp.expression for exp in used_change]
        used_behavior = Behavior.objects.filter(isUsed=1)
        used_behavior_exps = [exp.expression for exp in used_behavior]
        regexs = conbine_expression(role, used_behavior_exps, content, used_change_exps)

    else:
        pass

    # print(regexs)
    for line in regexs:
        [position_part, expression] = line.split('||')
        q = Regex(expression=expression, position_part=position_part)
        q.save()

def conbine_expression(role, behavior, middle, change):
    behavior_select = '(' + '|'.join(behavior) + ')'
    middle_select = '(' + '|'.join(middle) + ')'
    change_select = '(' + '|'.join(change) + ')'
    empty = '(\w+)'
    space = ' '
    global regex_strs
    if role == 'behavior':
        regex1 = 'middle:middle||' + empty + space + behavior_select + space + empty + space + change_select + space + empty
        regex2 = 'right:change||' + empty + space + behavior_select + space + middle_select + space + empty + space + empty
        regex3 = 'middle:middle||' + empty + space + change_select + space + empty + space + behavior_select + space + empty
        regex4 = 'left:change||' + empty + space + empty + space + middle_select + space + behavior_select + space + empty

        regex_strs = [
        '[behavior] + some_words + [change]',
        '[behavior] + [middle] + some_word',
        '[change] + some_words + [behavior]',
        'some_word + [middle] + [behavior]'
        ]

        return([regex1, regex2, regex3, regex4])

    elif role == 'change':
        regex1 = 'middle:middle||' + empty + space + change_select + space + empty + space + behavior_select + space + empty
        regex2 = 'right:behavior||' + empty + space + change_select + space + middle_select + space + empty + space + empty
        regex3 = 'middle:middle||' + empty + space + behavior_select + space + empty + space + change_select + space + empty
        regex4 = 'left:behavior||' + empty + space + empty + space + middle_select + space + change_select + space + empty
 
        regex_strs = [
        '[change] + some_words + [behavior]',
        '[change] + [middle] + some_word',
        '[behavior] + some_words + [change]',
        'some_word + [middle] + [change]'
        ]
        
        return([regex1, regex2, regex3, regex4])

    elif role == 'middle':
        regex1 = 'right:change||' + empty + space + behavior_select + space + middle_select + space + empty + space + empty
        regex2 = 'left:change||' + empty + space + empty + space + middle_select + space + behavior_select + space + empty
        regex3 = 'right:behavior||' + empty + space + change_select + space + middle_select + space + empty + space + empty
        regex4 = 'left:behavior||' + empty + space + empty + space + middle_select + space + change_select + space + empty

        regex_strs = [
        '[behavior] + [middle] + some_word',
        'some_word  + [middle] + [behavior]',
        '[change] + [middle] + some_word',
        'some_word + [middle] + [change]'
        ]
        
        return([regex1, regex2, regex3, regex4])

    else:
        pass    


def search_new_snippets(request):
    print('search_new_snippets')
    # f = open('./static/PMRCT/mining_data/clinical_evidence.mini','r')
    f = open('./static/PMRCT/mining_data/clinical_evidence.filtered','r')
    regexs = Regex.objects.all()
    snippets = []
    for line in f: 
        for regex in regexs:
            found  = re.findall(regex.expression, line)
            if len(found) > 0: 
                for s in found:
                    snippets.append(regex.position_part + '||' + ' '.join(s))
    f.close()  

    snippets_uniq = list(set(snippets))
    f = open('./static/PMRCT/mining_data/snippets','w')
    for expression in snippets_uniq:
        expression = expression.strip()
        f.write(expression + '\n')
    f.close()

    snippets_num = len(snippets_uniq)
    # print(snippets_uniq)
    data = {'snippets_num': snippets_num}
    return HttpResponse(json.dumps(data), content_type='application/json') 
    # print(snippets_uniq)

def find_unused_expressions():

    unused_change = Change.objects.filter(isUsed=0)    
    if len(unused_change) > 0:
        for exp in unused_change: 
            exp.isUsed=1
            exp.save()
        return {'role':'change', 'content':[exp.expression for exp in unused_change]}

    unused_behavior = Behavior.objects.filter(isUsed=0)    
    if len(unused_behavior) > 0:
        for exp in unused_behavior: 
            exp.isUsed=1
            exp.save()
        return {'role':'behavior', 'content':[exp.expression for exp in unused_behavior]}

    unused_middle = Middle.objects.filter(isUsed=0)    
    if len(unused_middle) > 0:
        for exp in unused_middle: 
            exp.isUsed=1
            exp.save()
        return {'role':'middle', 'content':[exp.expression for exp in unused_middle]}


def input_seed_data(request):
    print('input_seed_data')
    # print(os.path.dirname(os.path.abspath(__file__)))
    f = open('./static/PMRCT/seed_data/behavioral_expression.pos-neg-unknown','r') 

    items = TrainingData.objects.all()
    for item in items: item.delete()

    for line in f:
        line = line.replace('\n','') 
        line = line.strip()
        [isBehavior, expression] = line.split(':') 
        q = TrainingData(expression=expression, isBehavior=isBehavior)
        q.save()
    f.close()

    f = open('./static/PMRCT/seed_data/behavior.expression','r') 

    items = Behavior.objects.all()
    for item in items: item.delete()

    for line in f:
        line = line.replace('\n','') 
        line = line.strip()
        expression = line
        q = Behavior(expression=expression, isUsed=1)
        q.save()
    f.close()

    f = open('./static/PMRCT/seed_data/change.expression','r') 

    items = Change.objects.all()
    for item in items: item.delete()

    for line in f:
        line = line.replace('\n','') 
        line = line.strip()
        expression = line
        q = Change(expression=expression, isUsed=1)
        q.save()
    f.close()


    f = open('./static/PMRCT/seed_data/top_freq.middle','r') 

    items = Middle.objects.all()
    for item in items: item.delete()

    for line in f:
        line = line.replace('\n','') 
        line = line.strip()
        expression = line
        q = Middle(expression=expression, isUsed=0)
        q.save()
    f.close()    
    learn_TrainingData()

    data = {'message': 'seed data is loaded!'}
    return HttpResponse(json.dumps(data), content_type='application/json')  


def learn_TrainingData():
    exps = TrainingData.objects.all()
    train_X = [exp.expression for exp in exps]
    train_Y = [exp.isBehavior for exp in exps]

    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit(train_X)
    joblib.dump(vectorizer, 'TrainingData.vectorizer', compress=True)
    vectorizer = joblib.load('TrainingData.vectorizer')    
    train_x = vectorizer.transform(train_X)
    train_x = preprocessing.normalize(train_x, norm='l2')
    train_y = np.array(train_Y)
    print(train_x.shape)

    estimator = svm.LinearSVC(C=1000)
    multi_svm = OneVsRestClassifier(estimator)
    multi_svm.fit(train_x, train_y)
    joblib.dump(multi_svm, 'svc-TrainingData.model', compress=True)
    print('learn_TrainingData ended')


def snippets_classification(request):
    print(snippets_classification)
    f = open('./static/PMRCT/mining_data/snippets','r') 
    behaviorList = []
    changeList = []

    BehaviorInstances = Behavior.objects.all()
    BehaviorList = [ins.expression for ins in BehaviorInstances] 
    NewBehaviorList = []

    MiddleInstances = Middle.objects.all()
    MiddleList = [ins.expression for ins in MiddleInstances] 
    NewMiddleList = []

    ChangeInstances = Change.objects.all()
    ChangeList = [ins.expression for ins in ChangeInstances] 
    NewChangeList = []

    snippets = []
    positions = []
    parts = []
    for line in f:
        line = line.replace('\n','') 
        line = line.strip()
        [position_part, snippet] = line.split('||')
        [position, part] = position_part.split(':')
        positions.append(position)
        parts.append(part)
        snippets.append(snippet)
    f.close()

    vectorizer = joblib.load('TrainingData.vectorizer') 
    snippets_v = vectorizer.transform(snippets).toarray()
    classifier = joblib.load('svc-TrainingData.model')
    pred_snippets_dist = classifier.decision_function(snippets_v)
    positive_result = {}
    negative_result = {}
    unknown_result ={}
    # print(len(pred_snippets_dist))
    for i in range(len(pred_snippets_dist)):
        dist = pred_snippets_dist[i]
        label = classifier.classes_[dist.argmax()]
        max_dist = max(dist)
        # print(i, label, max_dist)
        if label == 0:
            negative_result[i] = max_dist
        elif label == 1:
            positive_result[i] = max_dist
        elif label == 2:
            unknown_result[i] = max_dist
        else:
            pass    

    # print(pred_snippets_dist)

    result = []
    for k in reversed(sorted(positive_result, key=lambda k:positive_result[k])): 
        snippet = snippets[k]
        words = snippet.split(' ')
        left = words[1]
        middle = words[2:-2]
        middle = ' '.join(middle)
        right = words[-2]
        # tokens = nltk.word_tokenize(snippets[k])
        # tags = nltk.pos_tag(tokens)
        # print(positive_result[k], k, snippets[k], positions[k], parts[k], left, middle, right)

        word = ''
        if positions[k] == 'left': 
            word = left
        elif positions[k] == 'middle':    
            word = middle
        elif positions[k] == 'right':
            word = right
        else:
            pass

        if filter(parts[k], word, BehaviorList, ChangeList, MiddleList) == 1:
            if parts[k] == 'behavior':
                if not word in NewBehaviorList: 
                    NewBehaviorList.append(word)
                    # print(positive_result[k], k, snippets[k], '-->', positions[k], parts[k])
            if parts[k] == 'change':
                if not word in NewChangeList: 
                    NewChangeList.append(word) 
                    # print(positive_result[k], k, snippets[k], '-->', positions[k], parts[k])              
            if parts[k] == 'middle':
                if not word in NewMiddleList: 
                    NewMiddleList.append(word)
                    # print(positive_result[k], k, snippets[k], '-->', positions[k], parts[k])


    print('New Behavior:' + str(len(NewBehaviorList)))
    print('New Change:' + str(len(NewChangeList)))
    print('New Middle:' + str(len(NewMiddleList)))


    data = {
    'positive_num': len(positive_result),
    'negative_num': len(negative_result),
    'unknown_num': len(unknown_result),
    'new_behavior': NewBehaviorList[:20],
    'new_change': NewChangeList[:20],
    # 'new_middle': NewMiddleList[:20]
    }

    for expression in NewMiddleList[:20]:
        q = Middle(expression=expression, isUsed=0)
        q.save()     

    nz = open('./static/PMRCT/mining_data/nearzero.positive', 'w')
    nzid = 0
    for k in sorted(positive_result, key=lambda k:positive_result[k]):
        if positive_result[k] <= 0: 
            continue
        else:
            print(k, snippets[k])
            nz.write(snippets[k] + '\n')
            nzid += 1
            if nzid >= 10: break
    nz.close()
    return HttpResponse(json.dumps(data), content_type='application/json')  


def filter(part, word, BehaviorList, ChangeList, MiddleList):
    # 1: not filter
    # 0: filter
    stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
    if word.lower() in stopwords: return 0
    if re.search(r'[A-Z0-9]+', word): return 0

    behavior_select = '(' + '|'.join(BehaviorList) + ')'
    middle_select = '(' + '|'.join(MiddleList) + ')'
    change_select = '(' + '|'.join(ChangeList) + ')'

    if part == 'behavior':
        if re.search(behavior_select, word): return 0
        if re.search(change_select, word): return 0
    elif part == 'middle':
        if re.search(middle_select, word): return 0  
    elif part == 'change':
        if re.search(change_select, word): return 0
        if re.search(behavior_select, word): return 0
    else:
        pass

    return 1


def register_new_phrases(request):
    print(register_new_phrases)
    phrase = request.POST['phrase']
    phrase_no = request.POST['phrase_no']

    if len(phrase) > 0:
        phraseList = phrase.split('||')    
        for expression in phraseList:
            q = TrainingData(expression=expression, isBehavior=1)
            q.save()       

    if len(phrase_no) > 0:
        phraseList = phrase_no.split('||')    
        for expression in phraseList:
            q = TrainingData(expression=expression, isBehavior=0)
            q.save()       

    data = {'snippets_num': 'register_new_words'}
    return HttpResponse(json.dumps(data), content_type='application/json')  

def register_new_words(request):
    print(register_new_words)
    behavior = request.POST['behavior']
    change = request.POST['change']
    middle = request.POST['middle']    
    behavior_no = request.POST['behavior_no']
    change_no = request.POST['change_no']
    middle_no = request.POST['middle_no']

    if len(behavior) > 0:
        behaviorList = behavior.split('||')    
        for expression in behaviorList:
            q = Behavior(expression=expression, isUsed=0)
            q.save()       

    if len(change) > 0:
        changeList = change.split('||')    
        for expression in changeList:
            q = Change(expression=expression, isUsed=0)
            q.save()   

    if len(middle) > 0:
        middleList = middle.split('||')    
        for expression in middleList:
            q = Middle(expression=expression, isUsed=0)
            q.save()   

    if len(behavior_no) > 0:
        behaviorList = behavior_no.split('||')    
        for expression in behaviorList:
            q = Behavior(expression=expression, isUsed=2)
            q.save()       

    if len(change_no) > 0:
        changeList = change_no.split('||')    
        for expression in changeList:
            q = Change(expression=expression, isUsed=2)
            q.save()   

    if len(middle_no) > 0:
        middleList = middle_no.split('||')    
        for expression in middleList:
            q = Middle(expression=expression, isUsed=2)
            q.save()  

    data = {'snippets_num': 'register_new_words'}
    return HttpResponse(json.dumps(data), content_type='application/json')  

def active_learning(request):
    print('active_learning')
    nz = open('./static/PMRCT/mining_data/nearzero.positive', 'r')
    nearzeros = []
    for line in nz:
        line = line.replace('\n','') 
        line = line.strip()
        nearzeros.append(line)
    data = {'nearzeros': '||'.join(nearzeros)}
    print('kuru2')
    return HttpResponse(json.dumps(data), content_type='application/json')     
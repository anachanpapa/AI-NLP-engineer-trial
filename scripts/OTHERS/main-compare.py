import os
import json
import codecs 
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
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
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn. model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from gensim import corpora 
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier

def CharBasedTrigram(element): 
    unigram = list(element)
    bigram = []
    trigram = []
    i = 0
    for i in range(len(unigram)-1):
        bigram.append(unigram[i]+unigram[i+1])
        i = i+1
    i = 0    
    for i in range(len(unigram)-2):
        trigram.append(unigram[i]+unigram[i+1]+unigram[i+2])
        i = i+1
    return bigram + trigram

def training():
    trainInput = open('../data/training_variants', 'r')
    trainInput.readline()
    trainPhases = open('../data/processed_data/training_phrases', 'r')

    train_X = []
    train_Y = []
    for item, phrase in zip(trainInput, trainPhases) :
        item = item.replace('\n','') 
        item = item.strip()
        phrase = phrase.replace('\n','') 
        phrase = phrase.strip()
        wordsInPhrase = phrase.split(' ')

        [id, gene, variaion, mclass] = item.split(',')
        train_Y.append(mclass)

        feature = [gene, variaion, variaion, variaion,]
        feature = feature + CharBasedTrigram(gene) 

        wordlist = ['Fusion', 'binding','Copy','deletion','Deletion','deletion/insertion','deletions','DNA','domain','Domain','duplications','Epigenetic','Hypermethylation','in','insertion','insertions','insertions/deletions','internal','Loss','missense','mutations','Mutations','Number','of','Promoter','Silencing','Splice','tandem','the','Transactivation','Truncating','Upstream','Variant']
        vlist = re.split(r'\s+|\_', variaion)
        for v in vlist:
            # print(variaion, v)
            feature.append(v)
            feature.append(v)
            feature.append(v)
            if not v in wordlist:
                feature = feature + CharBasedTrigram(v)

        # weight-->  feature : wordsInPhrase = 1: 1
        # howManyTimes = int(len(wordsInPhrase) / len(feature))
        # print(howManyTimes, end='-->')
        # pool = []
        # if howManyTimes >= 2:
        #     for i in range(howManyTimes-1):
        #         pool = pool + feature
        #     feature = pool
        # print(len(wordsInPhrase) / len(feature))

        feature = feature + wordsInPhrase
        # feature = wordsInPhrase

        train_X.append(' '.join(feature))

    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit(train_X)
    joblib.dump(vectorizer, 'TrainingData.vectorizer', compress=True)
    vectorizer = joblib.load('TrainingData.vectorizer')    
    train_x = vectorizer.transform(train_X)
    train_x = preprocessing.normalize(train_x, norm='l2')
    train_y = np.array(train_Y)
    print(train_x.shape)
    classifier_comparison(train_X, train_Y)


    # parameters = {
    #         'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
    #         'max_features'      : [3, 5, 10, 15, 20],
    #         'random_state'      : [0],
    #         'n_jobs'            : [1],
    #         'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
    #         'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    # }
    # clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
    # clf.fit(train_x, train_y)
    # print(clf.best_estimator_)

    # estimator = svm.LinearSVC(C=1000)
    # # estimator = svm.SVC()
    # estimator = LogisticRegression(C=100)
    # estimator.fit(train_x, train_y)

    # multi_svm = OneVsRestClassifier(estimator)
    # multi_svm.fit(train_x, train_y)

    joblib.dump(multi_svm, 'svc-TrainingData.model', compress=True)
    # joblib.dump(estimator, 'svc-TrainingData.model', compress=True)
    print('training completed')
    trainInput.close()
    trainPhases.close()

def classifier_comparison(train_X, train_Y):
        # vectorizer = CountVectorizer(ngram_range=(1, 3))
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        dicsize = len(train_X[0])
        vectorizer.fit(train_X)
        train_X = vectorizer.transform(train_X)
        train_X = preprocessing.normalize(train_X, norm='l2')

        # pca = PCA(n_components=50)
        # train_X = pca.fit_transform(train_X)

        # pca = PCA(n_components=500)
        # pca.fit(train_X)
        # train_X = pca.transform(train_X)

        print(train_X.shape)
        # train_X = vectorizer.transform(train_X).toarray()
        # vocabulary = vectorizer.get_feature_names()
        train_Y= np.array(train_Y)
        train_x, test_x, train_y, test_y  = train_test_split(train_X, train_Y, test_size=0.3)
        names = [
                 "Random Forest", 
                 "Linear SVM 1",
                 "Linear SVM 10",
                 "Linear SVM 100",
                 "Linear SVM 1000",
                 "LogisticRegression 1",
                 "LogisticRegression 10",
                 "LogisticRegression 100",
                 "LogisticRegression 1000",
                 "Nearest Neighbors",
                 "RBF SVM", 
                 "Decision Tree",
                 "AdaBoost"
                 ]

        classifiers = [
                # SCW1(C=1.0, ETA=1.0),
                # RandomForestClassifier(
                #     bootstrap=True, 
                #     class_weight=None, 
                #     criterion='gini',
                #     max_depth=100, 
                #     max_features=15, 
                #     max_leaf_nodes=None,
                #     min_impurity_decrease=0.0, 
                #     min_impurity_split=None,
                #     min_samples_leaf=1, 
                #     min_samples_split=15,
                #     min_weight_fraction_leaf=0.0, 
                #     n_estimators=10, 
                #     n_jobs=1,
                #     oob_score=False, 
                #     random_state=0, 
                #     verbose=0, 
                #     warm_start=False
                #     ),
                RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, min_samples_leaf=3),                
                svm.SVC(kernel="linear", C=1),
                svm.SVC(kernel="linear", C=10),
                svm.SVC(kernel="linear", C=100),
                svm.SVC(kernel="linear", C=1000),
                LogisticRegression(C=1),
                LogisticRegression(C=10),
                LogisticRegression(C=100),
                LogisticRegression(C=1000),
                KNeighborsClassifier(3),
                svm.SVC(kernel="rbf", gamma=0.0001, C=1000),
                DecisionTreeClassifier(max_depth=7),
                AdaBoostClassifier()]

        print('')
        print('--- Classfier Comparison ---')
        for name, clf in zip(names, classifiers):
                clf.fit(train_x, train_y)
                score = clf.score(test_x, test_y)
                print("name: {0} -- score: {1}".format(name, score))


def testing():
    trainInput = open('../data/training_variants', 'r')
    trainPhases = open('../data/processed_data/training_phrases', 'r')   
    trainInput.readline()
    train_X = []
    train_Y = []
    for item, phrase in zip(trainInput, trainPhases):
        item = item.replace('\n','') 
        item = item.strip()
        [id, gene, variaion, mclass] = item.split(',')
        train_Y.append(mclass)
        phrase = phrase.replace('\n','') 
        phrase = phrase.strip()
        wordsInPrase = phrase.split(' ')

        feature = [gene, variaion, variaion, variaion,]
        feature = feature + CharBasedTrigram(gene) 

        wordlist = ['Fusion', 'binding','Copy','deletion','Deletion','deletion/insertion','deletions','DNA','domain','Domain','duplications','Epigenetic','Hypermethylation','in','insertion','insertions','insertions/deletions','internal','Loss','missense','mutations','Mutations','Number','of','Promoter','Silencing','Splice','tandem','the','Transactivation','Truncating','Upstream','Variant']
        vlist = re.split(r'\s+|\_', variaion)
        for v in vlist:
            # print(variaion, v)
            feature.append(v)
            feature.append(v)
            feature.append(v)
            if not v in wordlist:
                feature = feature + CharBasedTrigram(v)

        feature = feature + wordsInPrase
        train_X.append(' '.join(feature))

    vectorizer = joblib.load('TrainingData.vectorizer') 
    train_X_v = vectorizer.transform(train_X).toarray()
    classifier = joblib.load('svc-TrainingData.model')


    clf = svm.LinearSVC(loss=GS_loss, C=GS_C)
    score = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

    # pred_dist = classifier.decision_function(train_X_v)
    # pred_Y = []
    # for i in range(len(pred_dist)):
    #     dist = pred_dist[i]
    #     label = classifier.classes_[dist.argmax()]
    #     pred_Y.append(label)

    # correct = 0
    # for (given, pred) in zip(train_Y, pred_Y):
    #     if given == pred: correct = correct + 1

    # print('Accuracy: ' + str(correct/len(pred_Y)) + ': ' + str(correct) + '/' + str(len(pred_Y)))

    trainInput.close()
    print('testing completed')  


def predicting():
    testInput = open('../data/test_variants', 'r')
    testInput.readline()
    testPhases = open('../data/processed_data/test_phrases.translated', 'r')
    submission = open('../data/submissionFile-sasaki', 'w')


    test_X = []
    test_Y = []
    for item, phrase in zip(testInput, testPhases):
        item = item.replace('\n','') 
        item = item.strip()
        [id, gene, variaion] = item.split(',')
        phrase = phrase.replace('\n','') 
        phrase = phrase.strip()
        wordsInPhrase = phrase.split(' ')

        feature = [gene, variaion, variaion, variaion,]
        feature = feature + CharBasedTrigram(gene) 

        wordlist = ['Fusion', 'binding','Copy','deletion','Deletion','deletion/insertion','deletions','DNA','domain','Domain','duplications','Epigenetic','Hypermethylation','in','insertion','insertions','insertions/deletions','internal','Loss','missense','mutations','Mutations','Number','of','Promoter','Silencing','Splice','tandem','the','Transactivation','Truncating','Upstream','Variant']
        vlist = re.split(r'\s+|\_', variaion)
        for v in vlist:
            # print(variaion, v)
            feature.append(v)
            feature.append(v)
            feature.append(v)
            if not v in wordlist:
                feature = feature + CharBasedTrigram(v)

        feature = feature + wordsInPhrase
        test_X.append(' '.join(feature))


    vectorizer = joblib.load('TrainingData.vectorizer') 
    # test_X_v = vectorizer.transform(test_X).toarray()
    test_X_v = vectorizer.transform(test_X)
    classifier = joblib.load('svc-TrainingData.model')
    pred_dist = classifier.decision_function(test_X_v)
    submission.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9' + '\n')
    for i in range(len(pred_dist)):
        dist = pred_dist[i]
        label = classifier.classes_[dist.argmax()]

        if label == '1': submission.write(str(i) + ',' + '1,0,0,0,0,0,0,0,0' + '\n')
        elif label == '2': submission.write(str(i) + ',' + '0,1,0,0,0,0,0,0,0' + '\n') 
        elif label == '3': submission.write(str(i) + ',' + '0,0,1,0,0,0,0,0,0' + '\n')
        elif label == '4': submission.write(str(i) + ',' + '0,0,0,1,0,0,0,0,0' + '\n')
        elif label == '5': submission.write(str(i) + ',' + '0,0,0,0,1,0,0,0,0' + '\n')
        elif label == '6': submission.write(str(i) + ',' + '0,0,0,0,0,1,0,0,0' + '\n')
        elif label == '7': submission.write(str(i) + ',' + '0,0,0,0,0,0,1,0,0' + '\n')
        elif label == '8': submission.write(str(i) + ',' + '0,0,0,0,0,0,0,1,0' + '\n')
        elif label == '9': submission.write(str(i) + ',' + '0,0,0,0,0,0,0,0,1' + '\n')
        else: print('Error')

    testInput.close()
    testPhases.close()
    submission.close()
    print('predicting completed')    

if __name__ == '__main__':
    training()
    # testing()
    predicting()


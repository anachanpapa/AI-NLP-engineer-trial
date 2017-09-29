import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from nltk import stem
import re


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


# conditions
TRANSLATE = 1
MINED_DATA = 0
VARIATION = 0
STEMMING = 1
VECTORIZER = 0 # 0: TFIDF 1:UNIGRAM 2:BIGRAM 3:TRIGRAM
DO_SVD = 1
GRID_SEARCH = 0

#Read Data
train_variant = pd.read_csv("../data/training_variants")
test_variant = pd.read_csv("../data/test_variants")

# select whether apply mined data
train_text = None
test_text = None
if MINED_DATA:
    train_text = pd.read_csv("../data/processed_data/training_phrases", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    test_text = pd.read_csv("../data/processed_data/test_phrases.translated", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
else:    
    train_text = pd.read_csv("../data/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    if TRANSLATE:
        test_text = pd.read_csv("../data/test_text.translated", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    else:
        test_text = pd.read_csv("../data/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])



train = pd.merge(train_variant, train_text, how='left', on='ID')
x_train = train.drop('Class', axis=1)
x_test = pd.merge(test_variant, test_text, how='left', on='ID')
test_index = x_test['ID'].values

data = np.concatenate((x_train, x_test), axis=0)
data=pd.DataFrame(data)
data.columns = ["ID", "Gene", "Variation", "Text"]


# Adding n-grams in variations
if VARIATION:
    print('start adding variation parts')
    stemmer = stem.PorterStemmer()
    stemdic = {}
    pool = []
    for line, did, variation in zip(data.Text, data.ID, data.Variation):
        vlist = re.split(r'\s+|\_', variation)
        vstrs = [variation, variation, variation]
        for v in vlist:
            vstrs.append(v)
            vstrs.append(v)
            vstrs.append(v)
            vstrs = vstrs + CharBasedTrigram(v) 
        vstr = ' '.join(vstrs)  
        line = vstr + ' ' + vstr + ' ' + vstr + ' ' + line  
        pool.append(line)
    data.Text = pool


# lower case, stemming and cleaning 
if STEMMING:
    print('start processing data')
    stemmer = stem.PorterStemmer()
    stemdic = {}
    pool = []
    for line in data.Text:
        stemmedList = []
        for word in line.split(' '):
            if word in stemdic: 
                stemmedList.append(stemdic[word])
            else:
                stemmed = stemmer.stem(word)
                stemmedList.append(stemmed) 
                stemdic[word] = stemmed
        pool.append(' '.join(stemmedList))       
    data.Text = pool

# vectorize
mod_TD = None
if VECTORIZER == 0: 
    print('start TFIDF')
    mod=TfidfVectorizer(min_df=5, max_features=500, stop_words='english')
    mod_TD=mod.fit_transform(data.Text)
elif VECTORIZER == 1:
    print('start UNIGRAM')
    mod=CountVectorizer(ngram_range=(1, 1))
    mod_TD=mod.fit_transform(data.Text)
elif VECTORIZER == 2:
    print('start BIGRAM')
    mod=CountVectorizer(ngram_range=(1, 2))
    mod_TD=mod.fit_transform(data.Text)
elif VECTORIZER == 3:
    print('start TRIGRAM')
    mod=CountVectorizer(ngram_range=(1, 3))
    mod_TD=mod.fit_transform(data.Text)


#SVD
if DO_SVD:
    print('start SVD')
    SVD=TruncatedSVD(200,random_state=40)
    SVD_FIT=SVD.fit_transform(mod_TD)
    mod_TD=pd.DataFrame(SVD_FIT)


encoder = LabelEncoder()
y_train = train['Class'].values
encoder.fit(y_train)
encoded_y = encoder.transform(y_train)

#GBM
print('start GBM')
gbm1 = None
model = GradientBoostingClassifier(max_features='sqrt', subsample=0.8, random_state=10)
if GRID_SEARCH:
    tuned_parameters = [
        {
        'learning_rate': [0.05, 0.1, 0.5], 
        'max_depth': [7, 10, 15],
        'min_samples_split': [300, 500, 800], 
        'min_samples_leaf': [10, 20, 50]
        }
    ]
    gscv = GridSearchCV(model, tuned_parameters, scoring="log_loss", verbose=10, n_jobs=-1, cv=4)
    gscv.fit(mod_TD[:3321], encoded_y)
    print('--------------------------')
    print("best score=", gscv.best_score_)
    print("best paramenters=", gscv.best_params_)
    gbm1 = gscv.best_estimator_

else:
    tuned_parameters = [
        {
        'learning_rate': [0.05], 
        'max_depth': [7],
        'min_samples_split': [800], 
        'min_samples_leaf': [50]
        }
    ]
    gscv = GridSearchCV(model, tuned_parameters, scoring="log_loss", verbose=10, n_jobs=-1, cv=4)
    gscv.fit(mod_TD[:3321], encoded_y)
    print('--------------------------')
    print("best score=", gscv.best_score_)
    # gbm1=GradientBoostingClassifier(learning_rate=0.05, min_samples_split=800,min_samples_leaf=50,max_depth=7,max_features='sqrt',subsample=0.8,random_state=10)
    gbm1=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
gbm1.fit(mod_TD[:3321],encoded_y)


#predictions
print('start predictions')
y_pred=gbm1.predict_proba(mod_TD[3321:])

#make submission file
subm_file = pd.DataFrame(y_pred)
subm_file['id'] = test_index
subm_file.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']

filename = 'submissionFile-sasaki-'
if TRANSLATE: filename += 'TRANSLATE'
if VARIATION: filename += 'VARIATION'
if STEMMING: filename += 'STEMMING'
# if VARIATION: filename += 'VARIATION'

subm_file.to_csv(filename, index=False)
print('completed')


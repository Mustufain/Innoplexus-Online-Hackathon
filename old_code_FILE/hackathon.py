import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
import nltk
import string
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.metrics import f1_score
import ast
from nltk.corpus import stopwords
from itertools import chain
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import LabelPowerset
from collections import defaultdict
from skmultilearn.adapt import MLkNN
import math
#only 4% records have non null values under full_text column in train data

nltk.download('punkt')
stop = stopwords.words('english')


def features(row):
    corpus = str(row).lower()
    corus = corpus.translate(None,string.punctuation)
    tokens = corpus.split(' ')
    hash = HashingVectorizer(n_features=20)
    X=hash.fit_transform(tokens)
    return X.toarray()[0]

def clean_ref_list(row):
    row = ast.literal_eval(row)
    return row

def count_authors(row):
        try:
            count=len(row.split(','))
            return count
        except Exception as e :
            return 0
info_train=pd.read_csv("information_train.csv",sep='\t')
train_raw=pd.read_csv("train.csv",sep=',')

train_raw['ref_list']=train_raw['ref_list'].apply(lambda row:clean_ref_list(row))
#lens = list(map(len, train_raw['ref_list']))
#train = pd.DataFrame({'pmid': np.repeat(train_raw['pmid'], lens),
 #                   'ref_list': list(chain.from_iterable(train_raw['ref_list']))})
#print len(train.ref_list.unique())

#train=pd.get_dummies(train_raw['ref_list'].apply(pd.Series).stack()).sum(level=0)
#train['pmid']=train_raw['pmid']
null_columns=info_train.columns[info_train.isnull().any()]
trainset=info_train.join(train_raw, lsuffix='_info_train', rsuffix='_train')

trainset['abstract'] = trainset['abstract'].apply(lambda row : features(row))
trainset['article_title'] = trainset['article_title'].apply(lambda row : features(row))
trainset['pub_date'] = pd.to_datetime(trainset['pub_date'])
trainset['year'] = trainset['pub_date'].dt.year
trainset['month'] = trainset['pub_date'].dt.month
trainset['day'] = trainset['pub_date'].dt.day
trainset['number_of_author'] = trainset['author_str'].apply(lambda row: count_authors(row))
trainset=trainset.drop(['author_str','pmid_train','pub_date','full_Text'],axis=1)
trainset = trainset.assign(**pd.DataFrame(trainset.abstract.values.tolist()).add_prefix('abstract_'))
trainset = trainset.assign(**pd.DataFrame(trainset.article_title.values.tolist()).add_prefix('articleTitle_'))
trainset= trainset.drop(['article_title','abstract'],axis=1)
Features = [col for col in trainset.columns if 'abstract' in col or 'article' in col]
y= trainset.ref_list.values
#print y
Labels = [col for col in trainset.columns if col.isdigit()]
Features.append("year")
Features.append("month")
Features.append("day")
Features.append("set")
Features.append('number_of_author')
X_train=np.array(trainset[Features])
#y=np.array(trainset[Labels])

#data = np.array(trainset)
#print data[0]
#trainset.to_csv("train_features.csv",sep='\t',index=False)
#trainset_X=trainset[['abstract','article_title','year','month','day','set']]
#trainset_Y=trainset[['ref_list']]
mlb = MultiLabelBinarizer()
y_enc = mlb.fit_transform(y)
labels=mlb.classes_

#print X.shape
#print Y.shape

#clf = OneVsRestClassifier(SVC(kernel='poly',C=1))

# X_train, X_test, y_train, y_test = train_test_split(
#          X, y_enc, test_size=0.3, random_state=0)
#
# classifier = LabelPowerset(GaussianNB())
# classifier.fit(X_train, y_train)
# prediction=classifier.predict(X_test)
# print f1_score(prediction,y_test,average='samples')



testset=pd.read_csv("information_test.csv",sep='\t')
test_raw=pd.read_csv("test.csv",sep=',')

testset['abstract'] = testset['abstract'].apply(lambda row : features(row))
testset['article_title'] = testset['article_title'].apply(lambda row : features(row))
testset['pub_date'] = pd.to_datetime(testset['pub_date'])
testset['year'] = testset['pub_date'].dt.year
testset['month'] = testset['pub_date'].dt.month
testset['day'] = testset['pub_date'].dt.day
testset['number_of_author'] = testset['author_str'].apply(lambda row: count_authors(row))
testset=testset.drop(['author_str','pub_date','full_Text'],axis=1)
testset = testset.assign(**pd.DataFrame(testset.abstract.values.tolist()).add_prefix('abstract_'))
testset = testset.assign(**pd.DataFrame(testset.article_title.values.tolist()).add_prefix('articleTitle_'))
testset= testset.drop(['article_title','abstract'],axis=1)
test_features = [col for col in testset.columns if 'abstract' in col or 'article' in col]
test_features.append("year")
test_features.append("month")
test_features.append("day")
test_features.append("set")
test_features.append("number_of_author")

X_test=np.array(testset[test_features])

classifier = OneVsRestClassifier(SVC(kernel='poly',C=1))
classifier.fit(X_train, y_enc)
prediction=classifier.predict(X_test)
a=prediction.toarray()
#print labels
#headerFile=np.insert(a,0,labels,axis=0)
np.savetxt('pred.csv',a,delimiter='\t')
#print testset.pmid.values

temp=pd.read_csv('pred.csv',delimiter='\t',names=labels)
rowmax = temp.max(axis=1)
temp.values == rowmax[:, None]
np.where(temp.values == rowmax[:,None])
import itertools as IT
import operator
idx = np.where(temp.values == rowmax[:,None])
groups = IT.groupby(zip(*idx), key=operator.itemgetter(0))
ref_list = [[temp.columns[j] for i, j in grp] for k, grp in groups]
submission = pd.DataFrame({'pmid': testset.pmid.values, 'ref_list': ref_list })
#submission["pmid,ref_list"] = submission["pmid"].map(str) + "," + submission["ref_list"].map(str)
#submission=submission[["pmid,ref_list"]]
submission.to_csv('submission.csv',sep=',',index=False)

exit(0)
#temp['pmid']=testset.pmid.values
temp['ref_list']=temp.idxmax(axis=1)
temp['pmid']=testset.pmid.values
temp=temp[['pmid','ref_list']]
#print temp.head()
exit(0)

# small test

print train_raw.head(1)
train=pd.get_dummies(train_raw['ref_list'].apply(pd.Series).stack()).sum(level=0)
rowmax = train.max(axis=1)
train.values == rowmax[:, None]
np.where(train.values == rowmax[:,None])
import itertools as IT
import operator
idx = np.where(train.values == rowmax[:,None])
groups = IT.groupby(zip(*idx), key=operator.itemgetter(0))
print [[train.columns[j] for i, j in grp] for k, grp in groups]
############
#submission = pd.DataFrame({'pmid': testset.pmid.values, labels: a})

# predict
#predictions = classifier.predict(X_test)
#clf.fit(X_train, y_train)
#y_true=clf.predict(X_test)

#print accuracy_score(y_test,y_true)
#print f1_score(predictions,y_test,average='samples')

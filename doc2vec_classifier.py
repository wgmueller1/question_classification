# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import string
from nltk.corpus import stopwords # Import the stop word list

#this is etl to remove stopwords
qlabel=[]
questions=[]
sentences=[]
with open('/Users/k25611/data/questions/train_5500.label.txt') as f:
    for line in f.readlines():
        tmp=line.split()
        qlabel.append(tmp[0])
        words = [w.lower() for w in tmp[1:]]# if not w.lower() in stopwords.words("english")]
        questions.append(words)
        
        
f.close()

#this formats the questions into the format required by gensim doc2vec
sentences = []
for i in range(len(questions)):
    string = "SENT_" + str(i)
    sentence = LabeledSentence(questions[i], (string,))
    sentences.append(sentence)
    
qlabels_top=map(lambda x: x.split(':')[0],qlabel)
#zip(qlabel,questions)
#vocabulary=' '.join(questions)
#vocabulary=set(map(lambda x: x.lower(),vocabulary.split()))

#sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

#fit the doc2vec model
model = Doc2Vec(size=100,alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

print len(model.docvecs)
print sentences[0]

#print the most similar document to question 1
print model.docvecs.most_similar('SENT_1')

print '\n'
print ' '.join(questions[0])
print ' '.join(questions[4356])
print ' '.join(questions[1104])
print ' '.join(questions[2891])

from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split,StratifiedKFold
import statsmodels as sm
import numpy as np
from sklearn.grid_search import GridSearchCV
from collections import Counter

qlabels_top=map(lambda x: x.split(':')[0],qlabel)
print dict(Counter(qlabels_top))
Y=np.argmax(sm.tools.tools.categorical(np.array(qlabels_top),drop=True),axis=1)
#splits the data in to train / test (model.docvecs are the document vectors)
x_train, x_test, y_train, y_test = train_test_split(np.array(model.docvecs), Y, test_size=0.1, random_state=42)

#set up the random forest
clf = RandomForestClassifier()
clf.fit(x_train,y_train)

y_predicted = clf.predict(x_test)

      
print "Classification report for classifier %s:\n%s\n" % (
    clf, metrics.classification_report(y_test, y_predicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_predicted)
print metrics.accuracy_score(y_test,y_predicted)

#set up the gridsearch for svm
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=y_test, n_folds=3)

#run the gridsearch and fit the best model
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(np.array(x_train), y_train)

print("The best classifier is: ", grid.best_estimator_)

y_predicted = grid.predict(x_train)


print "Classification report for classifier %s:\n%s\n" % (grid, metrics.classification_report(y_train, y_predicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, y_predicted)
print metrics.accuracy_score(y_train, y_predicted)

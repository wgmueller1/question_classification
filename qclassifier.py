import pandas as pd
import numpy as np
import string
import keras
import statsmodels as sm
from keras.preprocessing.text import Tokenizer
from itertools import chain
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

from sklearn.metrics import confusion_matrix
%pylab inline

def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f

labels=['ABBR:','DESC:','ENTY:','HUM:','LOC:']
qlabel=[]
questions=[]
with open('/Users/k25611/data/questions/train_5500.label.txt') as f:
    for line in f.readlines():
        tmp=line.split()
        qlabel.append(tmp[0])
        questions.append(' '.join(' '.join(tmp[1:]).translate(None, string.punctuation).strip().split()))
        
        
f.close()
qlabels_top=map(lambda x: x.split(':')[0],qlabel)
#zip(qlabel,questions)
vocabulary=' '.join(questions)
vocabulary=set(map(lambda x: x.lower(),vocabulary.split()))
data=keras.preprocessing.text.one_hot(' '.join(questions),n=500,filters=base_filter(),lower=True, split=" ")
lengths=map(lambda x: len(x.split(' ')),questions)
seq=[]
i=0
for l in lengths:
    seq.append(data[i:l+i])
    i=l+i

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

  
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(seq[0:4500], maxlen=maxlen)
X_test = sequence.pad_sequences(seq[4500:], maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

nb_classes=6
Y_train = sm.tools.categorical(np.array(qlabels_top[0:4500]),drop=True)
Y_test = sm.tools.categorical(np.array(qlabels_top[4500:]),drop=True)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print(questions[0])
print(seq[0])
print(X_train[0])

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256, input_length=maxlen))
model.add(LSTM(256))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))  


model.compile(loss='categorical_crossentropy', optimizer='adam')

history = model.fit(X_train, Y_train, nb_epoch=15, batch_size=250, verbose=1, show_accuracy=True, validation_split=0.25)
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])
pred = model.predict(X_test)

cm = confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(pred,axis=1))
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.show()

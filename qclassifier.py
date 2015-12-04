import pandas as pd
import string
import keras
from keras.preprocessing.text import Tokenizer
from itertools import chain
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1, JZS2, JZS3

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
data=keras.preprocessing.text.one_hot(' '.join(questions),n=len(vocabulary),filters=base_filter(),lower=True, split=" ")
lengths=map(lambda x: len(x.split(' ')),questions)
seq=[]
i=0
for l in lengths:
    seq.append(data[i:l+i])
    i=l+i

print("Vectorizing sequence data...")
tokenizer = Tokenizer(nb_words=len(vocabulary))
X_train = tokenizer.sequences_to_matrix(seq[0:4500], mode="binary")
X_test = tokenizer.sequences_to_matrix(seq[4500:], mode="binary")
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

nb_classes=6
Y_train = sm.tools.categorical(np.array(qlabels_top[0:4500]),drop=True)
print(Y_train)
Y_test = sm.tools.categorical(np.array(qlabels_top[4500:]),drop=True)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print("Building model...")
model = Sequential()
model.add(Dense(512, input_shape=(8603,)))
model.add(Activation('relu'))
#model.add(LSTM(512, init='glorot_uniform', inner_init='orthogonal',
#               activation='tanh', inner_activation='hard_sigmoid', 
#               return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

history = model.fit(X_train, Y_train, nb_epoch=10, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])

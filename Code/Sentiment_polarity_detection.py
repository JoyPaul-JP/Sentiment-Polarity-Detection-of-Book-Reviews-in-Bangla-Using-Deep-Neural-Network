import tensorflow as tf
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer



seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

!pip install bnlp_toolkit

print(tf.keras.__version__)
print(tf.__version__)
!python --version

from google.colab import drive
drive.mount('/content/drive')

movie_reviews = pd.read_excel(r"/content/drive/My Drive/Thesis/bookReviews_2000.xlsx")

movie_reviews.isnull().values.any()

movie_reviews.shape

movie_reviews.head()


import seaborn as sns
sns_plot=sns.countplot(x='Sentiment', data=movie_reviews)
fig = sns_plot.get_figure()
fig.savefig("data_shape.png")


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

Z=[]
Sent=[]
sentence_tokens_list = []
word_tokens_list = []
pos_tags_list = []
sentences = []

sentence_tokens_list.clear()
word_tokens_list.clear()
pos_tags_list.clear()
sentences.clear()


sentence = list(movie_reviews['Review'])

for sen in sentence:
    sentences.append(preprocess_text(sen))

print(sentences[:5])


Z.clear()

Sent.clear()

import nltk
nltk.download('punkt')
Z.clear()
local_pos_tags = []
local_pos_tags.clear()
pos_tags_list.clear()

import copy ## to implement shallow copy. See copy in python

from bnlp import NLTKTokenizer
from bnlp import POS
bn_pos = POS()
model_path = "/content/drive/My Drive/Thesis/bn_pos_model.pkl"
for i in range(len(sentences)):
    text = sentences[i]
    bnltk = NLTKTokenizer()
    
    
    word_tokens = bnltk.word_tokenize(text)
    word_tokens_copy = copy.copy(word_tokens)
    
    
    sentence_tokens = bnltk.sentence_tokenize(text)
    sentence_tokens_list.append(sentence_tokens)
    
    for j in range(len(word_tokens)):
        word = word_tokens[j]    
        pos_tags = bn_pos.tag(model_path, word)
        
        if(pos_tags[0][1] == "PU"):
            word_tokens_copy.remove(str(word))

            
    word_tokens_list.append(word_tokens_copy)



print("\n Sentence Tokens")
print(sentence_tokens_list[:3])

print("\n Word Tokens")
print(word_tokens_list[:3])

## We are creating word tokens first in 'word_tokens' . Then a shallow copy is made of 'word_tokens' to 
## 'word_tokens_copy'. We then check the POS tag of the tokens and if any one of them is "PU" (i.e punctuation)
## then we remove it from 'word_tokens_copy'. After iterating through all the tokens (hence removing all the "PU")
## we add 'word_tokens_copy' to 'word_tokens_list' .

## All these mess is done to remove the error 'list index out of range' while removing an element

copy_of_word_tokens_list = [] ## It is costly to compute "word_tokens_list".
                            ## So I'm keeping an copy so that I can experiment with one and change it as wish
copy_of_word_tokens_list.clear()
copy_of_word_tokens_list = copy.copy(word_tokens_list)

word_tokens_list = copy.copy(copy_of_word_tokens_list)

remove_counter = 0  ## to check if any array is removed for being empty
token_lngth = 0
token_length = len(word_tokens_list) ## to iterate over the whole "word_tokens_list" array
empty_positions = [] ## keep track of empty positions
empty_positions.clear()
print(token_length) ## How many positions are empty actually 

for i in range(token_length):
    if(len(copy_of_word_tokens_list[i]) == 0):
        word_tokens_list.pop(i)
        empty_positions.append(i)
        remove_counter += 1
        print("a")
        
        
print(remove_counter)

print(len(word_tokens_list))

print(len(sentence_tokens_list))

print(len(word_tokens_list))

print(len(sentence_tokens_list))


#word2vec
import json
import os
import re
import string
import numpy as np

from gensim.models import Word2Vec

model1 =Word2Vec(word_tokens_list, size=100, window=15, min_count=1,sg=0)
#model2 =Word2Vec.load('bn_w2v_model.text')
model1.wv.save_word2vec_format("/content/drive/My Drive/Thesis/myword2vec_model.txt")

model1["ভালো"]


y = movie_reviews['Sentiment']

print(len(y))
for i in range(len(empty_positions)):
    y.pop(empty_positions[i])

print(len(y))

X_train, X_test, y_train, y_test = train_test_split(word_tokens_list, y, test_size=0.10, random_state=42)

print(X_train[:5])
print(y_train[:5])

print(len(X_train))

len(X_test)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train[:5]

vocab_size = len(tokenizer.word_index) + 1


maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

X_train[:5]

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('/content/drive/My Drive/Thesis/myword2vec_model.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
embedding_matrix[10]


from keras import Sequential, optimizers
from keras.layers import Embedding, LSTM, Dense, Dropout,Bidirectional,ConvLSTM2D,GRU,SimpleRNN,RNN
from keras import regularizers

model2 = Sequential()
embedding_layer = Embedding(vocab_size, 100,weights=[embedding_matrix], input_length=maxlen , trainable=True)
model2.add(embedding_layer)

model2.add (Bidirectional(LSTM(128,  return_sequences=True,kernel_regularizer=regularizers.l2(0.15))))

model2.add(Conv1D(32, 4, activation='relu'))
model2.add (GRU(32,return_sequences=True, activation = 'tanh') )
model2.add(Conv1D(16, 4, activation='relu'))
model2.add(GlobalMaxPooling1D())

model2.add (Dropout(0.4))

model2.add(Dense(16,activation='relu'))


model2.add(Dense(1, activation='sigmoid'))


opt = optimizers.Adam(learning_rate=0.0005)
#lr=0.0005
model2.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

model2.summary()

history = model2.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.10)

score = model2.evaluate(X_test, y_test, verbose=1)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc = 'upper left')
plt.savefig('model_Accuracy.png')

plt.show()


plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc = 'upper left')
plt.savefig('model_loss.png')

plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score



yhat_classes = model2.predict_classes(X_test, verbose=1)
yhat_classes = yhat_classes[:, 0]
accuracy = accuracy_score(y_test, yhat_classes)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)


# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)





# confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, xticklabels='01', yticklabels='01', annot=True, fmt='g')


plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confussion_matrix.png')

plt.show()

r_fpr, r_tpr, _ = roc_curve(y_test,yhat_classes)
rf = roc_auc_score(y_test,yhat_classes)
plt.plot(r_fpr,r_tpr, linestyle ='-' %(rf) )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

fpr, tpr, _ = roc_curve(y_test,yhat_classes)
# plt.figure(figsize=(14,8))
plt.plot(fpr, tpr, color="red")
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.savefig('roc_curve.png')

plt.show()

# -*- coding: utf-8 -*-
# Получаем векторное представление предложений через упаковку из с помощью rnn автоэнкодер.


from __future__ import print_function
from __future__ import division  # for python2 compatibility

import pandas as pd
from collections import Counter
import tqdm
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import functools
import sys
import math
from Segmenter import Segmenter
from Tokenizer import Tokenizer
import TextNormalizer
import gensim
import scipy.spatial.distance
import codecs

from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers.core import Activation
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping


MAX_LEN = 50
rnn_size = 64
add_vectors = True

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# --------------------------------------------------------------------------

#word_dims = 32

print( 'Loading the w2v model...' )
w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/eek/polygon/w2v/w2v.CBOW=1_WIN=20_DIM=32.txt', binary=False)
word_dims = len(w2v.syn0[0])
print('Word vector length={0}'.format(word_dims))

# ------------------------------------------------------------------------


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s

# ------------------------------------------------------------------------

sent2id = dict()
sent_list = list()
word2id = dict()
word2id[u''] = 0

def store_sent(sent_str):
    words = TextNormalizer.tokenize_lemmas(sent_str)
    for word in words:
        if word not in word2id:
            word2id[word] = len(word2id)

    ids = tuple( [word2id[word] for word in words] )
    if ids not in sent2id:
        sent2id[ u' '.join(words) ] = len(sent_list)
        sent_list.append(ids)



segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Extracting sentences from " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        sent_id = store_sent(question)

        for parag_sent in segmenter.split(paragraph):
            sent_id = store_sent(parag_sent)

nb_sent = len(sent_list)
nb_words = len(word2id)

print('{} sentences'.format(nb_sent))
print('{} words'.format(nb_words))

max_len = min( MAX_LEN, max( [len(s) for s in sent_list] ) )
print('max_len={}'.format(max_len))


id2word = dict([(i,word) for (word,i) in word2id.iteritems() ])

# --------------------------------------------------------------------

nb_sent = len(sent_list)
X_train = np.zeros( (nb_sent, max_len), dtype='int32' )
y_train = np.zeros( (nb_sent, max_len, word_dims) )

for i,sent in enumerate(sent_list):
    for j,word_id in enumerate(sent[:max_len]):
        X_train[ i, j ] = word_id
        word = id2word[word_id]
        if word in w2v:
            y_train[i,j,:] = w2v[word]

#y_train = np.expand_dims(X_train, -1)

# --------------------------------------------------------------------



print('Building the network...')

input = Input(shape=(max_len,), dtype='int32')

embedding = Embedding(output_dim=word_dims,
                            input_dim=nb_words,
                            input_length=max_len,
                            #weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=True)

encoder = embedding(input)
encoder = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))(encoder)

decoder = RepeatVector(max_len)(encoder)
decoder = LSTM(word_dims, return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(word_dims, activation='tanh'))(decoder)

model = Model(inputs=input, outputs=decoder)
model.compile(loss='mse', optimizer='rmsprop')

if True:
    print('Start training...')

    model_checkpoint = ModelCheckpoint('rnn_ae.model', monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    history = model.fit(x=X_train,
                        y=y_train,
                        validation_split=0.2,
                        batch_size=256,
                        epochs=100,
                        verbose=1,
                        callbacks=[model_checkpoint, early_stopping])


model.load_weights('rnn_ae.model')

# --------------------------------------------------------------------

model = Model(inputs=input, outputs=encoder)
model.compile(loss='mse', optimizer='rmsprop')

for name, df in [('train', dftrain), ('test', dftest)]:

    quest_tuples = []
    parag_tuples = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Transforming question and sentences in '" + name+"'"):
        question = TextNormalizer.preprocess_question_str(row.question)
        question = u' '.join( TextNormalizer.tokenize_lemmas(question) )
        quest_tuples.append( sent_list[sent2id[question]] )

        quest_terms = TextNormalizer.tokenize_crops(question)

        paragraph = row.paragraph
        best_parag_sent = None
        max_match = 0.0
        for parag_sent in segmenter.split(paragraph):
            parag_terms = TextNormalizer.tokenize_crops(parag_sent)
            terms_match = len(set(quest_terms) & set(parag_terms))
            if terms_match>max_match:
                max_match = terms_match
                best_parag_sent = u' '.join( TextNormalizer.tokenize_lemmas(parag_sent) )

        parag_tuples.append( sent_list[sent2id[best_parag_sent]]  )


    nrow = len(quest_tuples)
    X_quest = np.zeros( (nrow,max_len), dtype='int32' )
    for i,sent_data in enumerate(quest_tuples):
        for j,term_id in enumerate(sent_data[:max_len]):
            X_quest[i,j] = term_id

    Z_quest = model.predict(X_quest)

    X_parag = np.zeros( (nrow, max_len), dtype='int32' )
    for i, sent_data in enumerate(parag_tuples):
        for j, term_id in enumerate(sent_data[:max_len]):
            X_parag[i, j] = term_id

    Z_parag = model.predict(X_parag)

    for i in tqdm.tqdm(range(nrow), total=nrow, desc="Writing rnn_cos to '" + name+"' dataframe"):
        v_quest = Z_quest[i]
        v_parag = Z_parag[i]
        c = v_cosine(v_quest, v_parag)
        df.loc[i, 'rnn_cos'] = c

    if add_vectors:
        quest_column_data = np.zeros(nrow)
        parag_column_data = np.zeros(nrow)
        for icol in tqdm.tqdm(range(rnn_size), total=rnn_size, desc="Writing vectors to '" + name + "' dataframe"):
            for jrow in range(nrow):
                quest_column_data[jrow] = Z_quest[jrow][icol]
                parag_column_data[jrow] = Z_parag[jrow][icol]

            df['quest_rnn_{}'.format(icol)] = quest_column_data
            df['parag_rnn_{}'.format(icol)] = parag_column_data

# --------------------------------------------------------------------

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

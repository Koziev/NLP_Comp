# -*- coding: utf-8 -*-
'''
Deep learning решение задачи конкурса https://contest.sdsj.ru/dashboard?problem=A 
(c) Козиев Илья inkoziev@gmail.com

Сеточная модель с несколькими входами.
Текст параграфа и вопроса упаковывается отдельними рекуррентными фрагментами графа, затем
результаты упаквки объединяются и обрабатываются финальными слоями классификации.


'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
from collections import Counter
import tqdm
import re
import numpy as np
import functools
import sys
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import scipy.spatial.distance
import gc
import gensim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input, Permute, Flatten, Reshape
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Lambda
from keras import backend as K
import keras.layers
from keras.layers.merge import concatenate, add, multiply

from keras.models import Model
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import Bidirectional
from keras.layers import concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from Segmenter import Segmenter
from Tokenizer import Tokenizer
import TextNormalizer
import Abbrev


# -------------------------------------------------------------------

max_paragraph_len = 100
max_question_len = 100  #50
max_len = 100

use_cnn = True
use_rnn = True

# --------------------------------------------------------------------------

#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=1_WIN=20_DIM=32.txt'
w2v_path = '/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt'
print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
vec_len = len(w2v.syn0[0])
print('Vector length={0}'.format(vec_len))

# ------------------------------------------------------------------------------

dftrain = pd.read_csv("../input/train_task1_latest.csv", encoding='utf-8')
dftest = pd.read_csv("../input/sdsj_A_test.csv", encoding='utf-8')

# ---------------------------------------------------------------------


columns = ['lsa_cos', 'w2v_cos',
           'len_paragraph',
           'len_question',
           'len_intersection',
           'idf_question',
           'idf_paragraph',
           'idf_intersection',
           'len_paragraph2',
           'len_question2',
           'len_intersection2',
           'idf_question2', 'idf_paragraph2', 'idf_intersection2',
           'best_sent_match',
           'max_w2v_cos',
           'avg_w2v_cos',
           'max_idf_intersection2_sent',
           'quest_ne_count',
           'nes_intersection',
           #'best_ne_match',
           #'quest_fw_count',
           #'fw_intersection',
           #'best_fw_match',
           'sum_w0',
           'sum_w1',
           'max_tail_match',
           'max_lsa_per_sent',
           'min_lsa_per_sent',
           'avg_lsa_per_sent',
           #'quest_oov_count',
           'parag_oov_count', 'oov_intersection', # features_18_1.py
           'max_intersect3',  # features_18_3.py
           'max_syn_sim',  # features_18_4.py
           'rnn_cos',
           'min_wmd',
           'max_wmd',
           'avg_wmd',
           'wmd',
           'max_fuzz_partial_ratio_str',
           ]

#LSA_DIMS = 100
#columns.extend( ['lsa_quest_{}'.format(i) for i in range(LSA_DIMS)] )
#columns.extend( ['lsa_parag_{}'.format(i) for i in range(LSA_DIMS)] )


dftrain_xgb = pd.read_csv("../data/dftrain.csv", encoding='utf-8')[columns]
dftest_xgb = pd.read_csv("../data/dftest.csv", encoding='utf-8')[columns]

dftest_xgb = dftest_xgb.replace([np.inf, -np.inf], np.nan).fillna(1000.0)

X_xgb_train = dftrain_xgb.values
X_xgb_test = dftest_xgb.values


#z = np.where(np.isnan(X_xgb_test))


scaler = MinMaxScaler()
scaler.fit(X_xgb_train)
X_xgb_train = scaler.transform(X_xgb_train)
X_xgb_test = scaler.transform(X_xgb_test)


# -------------------------------------------------------------------

X_parag_train = None
X_parag1_train = None
X_quest_train = None

X_parag_test = None
X_parag1_test = None
X_quest_test = None

y_train = dftrain['target'].values

word2id = dict()
word2id[u''] = 0

segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:

    X_parag  = np.zeros((df.shape[0], max_len), dtype=np.int32)
    X_parag1 = np.zeros((df.shape[0], max_len), dtype=np.int32)
    X_quest  = np.zeros((df.shape[0], max_len), dtype=np.int32)

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_words(question)[0:max_len]
        parag_words = TextNormalizer.tokenize_words(paragraph)[0:max_len]

        for word_pos,word in enumerate(quest_words):
            if word not in word2id:
                word2id[word] = len(word2id)

            X_quest[index, word_pos ] = word2id[word]

        for word_pos, word in enumerate(parag_words):
            if word not in word2id:
                word2id[word] = len(word2id)

            X_parag[index, word_pos] = word2id[word]


        quest_stems = TextNormalizer.tokenize_stems( question )
        quest_set = set(quest_stems)
        denom = float(len( quest_set ))

        max_intersect3 = 0.0
        best_sent = u''

        for parag_sent in segmenter.split(paragraph):
            parag_stems = TextNormalizer.tokenize_stems( Abbrev.normalize_abbrev(parag_sent) )
            intersect3 = len( quest_set & set(parag_stems) ) / denom
            if intersect3>max_intersect3:
                max_intersect3 = intersect3
                best_sent = parag_sent


        parag_words = TextNormalizer.tokenize_words(best_sent)
        for word_pos, word in enumerate(parag_words[0:max_len]):
            if word not in word2id:
                word2id[word] = len(word2id)

            X_parag1[index, word_pos] = word2id[word]


    if name=='train':
        X_parag_train = X_parag
        X_parag1_train = X_parag1
        X_quest_train = X_quest
    else:
        X_parag_test = X_parag
        X_parag1_test = X_parag1
        X_quest_test = X_quest

print('word2id.count={}'.format(len(word2id)))

# -------------------------------------------------------------------

print('Constructing the NN model...')

nb_words = len(word2id)
word_dims = vec_len
embedding_matrix = np.zeros( (nb_words, word_dims) )
for word,i in word2id.iteritems():
    if word in w2v:
        embedding_matrix[i,:] = w2v[word]

embedding = Embedding(output_dim=word_dims,
                            input_dim=nb_words,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            trainable=False)



nb_filters = 128
rnn_size = vec_len

final_merge_size = 0

# --------------------------------------------------------------------------------

input_parag = Input(shape=(max_paragraph_len,), dtype='int32')
parag_net = embedding(input_parag)

conv_list = []
merged_size = 0

if use_cnn:
    for kernel_size in range(2, 5):
        conv_layer = Conv1D(filters=nb_filters,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1)(parag_net)
        conv_layer = GlobalMaxPooling1D()(conv_layer)
        conv_list.append(conv_layer)
        merged_size += nb_filters

if use_rnn:
    encoder_rnn = Bidirectional(recurrent.LSTM(rnn_size, input_shape=(max_len, vec_len),
                                               return_sequences=False))(parag_net)
    merged_size += rnn_size*2
    #encoder_rnn = recurrent.LSTM(rnn_size, return_sequences=False)(parag_net)
    #merged_size += rnn_size
    conv_list.append(encoder_rnn)

if len(conv_list)>1:
    parag_merged = keras.layers.concatenate(inputs=conv_list)
    parag_encoder = Dense(units=int(merged_size), activation='relu')(parag_merged)
    final_merge_size += merged_size
else:
    parag_encoder = Dense(units=int(merged_size), activation='relu')(conv_list[0])
    final_merge_size += merged_size


# --------------------------------------------------------------------------------

input_parag1 = Input(shape=(max_len,), dtype='int32')
parag1_net = embedding(input_parag1)

conv_list = []
merged_size = 0

if use_cnn:
    for kernel_size in range(2, 5):
        conv_layer = Conv1D(filters=nb_filters,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1)(parag1_net)
        conv_layer = GlobalMaxPooling1D()(conv_layer)
        conv_list.append(conv_layer)
        merged_size += nb_filters

if use_rnn:
    encoder_rnn = Bidirectional(recurrent.LSTM(rnn_size, input_shape=(max_len, vec_len),
                                               return_sequences=False))(parag1_net)
    merged_size += rnn_size*2
    conv_list.append(encoder_rnn)

if len(conv_list)>1:
    parag_merged = keras.layers.concatenate(inputs=conv_list)
    parag1_encoder = Dense(units=int(merged_size), activation='relu')(parag_merged)
    final_merge_size += merged_size
else:
    parag1_encoder = Dense(units=int(merged_size), activation='relu')(conv_list[0])
    final_merge_size += merged_size

# -----------------------------------------------------------------------------------

input_quest = Input(shape=(max_len,), dtype='int32')
quest_net = embedding(input_quest)

conv_list = []
merged_size = 0

if use_cnn:
    for kernel_size in range(2, 5):
        conv_layer = Conv1D(filters=nb_filters,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1)(parag_net)
        conv_layer = GlobalMaxPooling1D()(quest_net)
        conv_list.append(conv_layer)
        merged_size += nb_filters

if use_rnn:
    encoder_rnn = Bidirectional(recurrent.LSTM(rnn_size, input_shape=(max_len, vec_len),
                                               return_sequences=False))(quest_net)
    merged_size += rnn_size*2
    #encoder_rnn = recurrent.LSTM(rnn_size, return_sequences=False)(quest_net)
    #merged_size += rnn_size
    conv_list.append(encoder_rnn)

if len(conv_list)>1:
    quest_merged = keras.layers.concatenate(inputs=conv_list)
    quest_encoder = Dense(units=int(merged_size), activation='relu')(quest_merged)
    final_merge_size += merged_size
else:
    quest_encoder = Dense(units=int(merged_size), activation='relu')(conv_list[0])
    final_merge_size += merged_size


# --------------------------------------------------------------------------------

nb_xgb_features = X_xgb_train.shape[1]
input_xgb = Input(shape=(nb_xgb_features,), dtype='float32')
xgb_encoder = Dense(units=nb_xgb_features, activation='relu')(input_xgb)

# --------------------------------------------------------------------------------

final = keras.layers.concatenate(inputs=[parag_encoder, parag1_encoder, quest_encoder, xgb_encoder])
#final = Dense(units=int(final_merge_size/2), activation='sigmoid')(final)
final = Dense(units=int(final_merge_size/2), activation='relu')(final)
final = Dense(units=int(final_merge_size/4), activation='relu')(final)
final = Dense(units=int(final_merge_size/8), activation='relu')(final)
final = Dense(units=1, activation='sigmoid', name='output')(final)
model = Model(inputs=[input_parag, input_parag1, input_quest, input_xgb], outputs=final)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

# --------------------------------------------------------------------------------

print('Start training...')

model_checkpoint = ModelCheckpoint('nn.model', monitor='val_acc',
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')


SEED = 123456
TEST_SHARE = 0.2
X_parag_train1, X_parag_val, y_train1, y_val = train_test_split( X_parag_train, y_train, test_size=TEST_SHARE, random_state=SEED )
X_parag1_train1, X_parag1_val, _, _ = train_test_split( X_parag1_train, y_train, test_size=TEST_SHARE, random_state=SEED )
X_quest_train1, X_quest_val, _, _ = train_test_split( X_quest_train, y_train, test_size=TEST_SHARE, random_state=SEED )
X_xgb_train1, X_xgb_val, _, y_val_ = train_test_split( X_xgb_train, y_train, test_size=TEST_SHARE, random_state=SEED )


history = model.fit(x=[X_parag_train1, X_parag1_train1, X_quest_train1, X_xgb_train1],
                    y=y_train1,
                    validation_data=([X_parag_val, X_parag1_val, X_quest_val, X_xgb_val], y_val),
                    batch_size=256,
                    epochs=100,
                    callbacks=[model_checkpoint, early_stopping])

print('Compute the submission...')
model.load_weights('nn.model')
y_submission = model.predict([X_parag_test, X_parag1_test, X_quest_test, X_xgb_test])[:, 0]

dftest['prediction'] = y_submission

dftest[['paragraph_id', 'question_id', 'prediction']].to_csv("../submit/nn5.csv", index=False)


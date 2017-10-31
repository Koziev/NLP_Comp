# -*- coding: utf-8 -*-
# coding: utf-8
"""
Фичи на базе w2v: косинус средних векторов текста и вопроса, максимальный/минимальный/средний косинус
средних векторов для вопроса и ближайшего предложения в тексте.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
from collections import Counter
import tqdm
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import functools
import sys
import gensim
import codecs
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import scipy.spatial.distance
import gc
import collections
from nltk.stem.snowball import RussianStemmer
from Segmenter import Segmenter
from Tokenizer import Tokenizer
import TextNormalizer

# ---------------------------------------------------------------------------------

print('Loading the w2v model...')
# w2v = gensim.models.KeyedVectors.load_word2vec_format(r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt', binary=False)
#w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/eek/polygon/w2v/w2v.CBOW=1_WIN=20_DIM=32.txt', binary=False)
w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt', binary=False)
vec_len = len(w2v.syn0[0])
print('Vector length={0}'.format(vec_len))

print('Loading datasets...')
dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# ---------------------------------------------------------------------------------


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s


def get_w2v(words, stems, veclen, w2v, stem2weight):
    v = np.zeros(veclen)
    denom = 0
    for (word, stem) in zip(words, stems):
        if word in w2v:
            denom += stem2weight[stem]
            v += np.asarray(w2v[word]) * stem2weight[stem]
    return v / denom if denom > 0 else v

# -------------------------------------------------------------------------

stem2freq = collections.Counter()
for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Counting word frequencies in " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph
        stem2freq.update( TextNormalizer.tokenize_stems(question) )
        stem2freq.update(TextNormalizer.tokenize_stems(paragraph))

total_freq = sum( stem2freq.values() )
stem2weight = dict( [ (w, math.log(total_freq/freq)) for (w,freq) in stem2freq.iteritems() ] )

# -----------------------------------------------------------------------------------

segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Build w2v features for " + name):
        quest_str = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_words(quest_str)
        parag_words = TextNormalizer.tokenize_words(paragraph)

        quest_stems = TextNormalizer.tokenize_stems(quest_str)
        parag_stems = TextNormalizer.tokenize_stems(paragraph)

        v1 = get_w2v(quest_words, quest_stems, vec_len, w2v, stem2weight)
        v2 = get_w2v(parag_words, parag_stems, vec_len, w2v, stem2weight)
        c = v_cosine(v1, v2)
        df.loc[index, 'w2v_cos'] = c

        max_w2v_cos = -1e38
        min_w2v_cos = 1e38
        avg_w2v_cos_num = 0.0
        avg_w2v_cos_denom = 1e-8

        for parag_sent in segmenter.split(paragraph):
            parag_words = TextNormalizer.tokenize_words(parag_sent)
            parag_stems = TextNormalizer.tokenize_stems(parag_sent)
            v2 = get_w2v(parag_words, parag_stems, vec_len, w2v, stem2weight)
            c = v_cosine(v1, v2)
            max_w2v_cos = max( max_w2v_cos, c )
            min_w2v_cos = min( min_w2v_cos, c )
            avg_w2v_cos_num += c
            avg_w2v_cos_denom += 1.0

        df.loc[index, 'max_w2v_cos'] = max_w2v_cos
        df.loc[index, 'min_w2v_cos'] = min_w2v_cos
        df.loc[index, 'avg_w2v_cos'] = avg_w2v_cos_num/avg_w2v_cos_denom


print('Storing modified datasets...')
dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

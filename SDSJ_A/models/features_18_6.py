# -*- coding: utf-8 -*-
"""
Word mover distance
"""


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

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# ------------------------------------------------------------------------

print('Loading the w2v model...')
w2v = gensim.models.KeyedVectors.load_word2vec_format(r'/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt', binary=False)
#w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/eek/polygon/w2v/w2v.CBOW=1_WIN=20_DIM=32.txt', binary=False)
vec_len = len(w2v.syn0[0])
print('Vector length={0}'.format(vec_len))

# ------------------------------------------------------------------------


segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Calculating word mover distance for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_words(question)
        parag_words = TextNormalizer.tokenize_words(paragraph)
        wmd = w2v.wmdistance(quest_words, parag_words)
        df.loc[index, 'wmd'] = wmd

        dist_sum = 0.0
        dist_denom = 1e-8
        min_dist = 1e38
        max_dist =  -1e38

        for parag_sent in segmenter.split(paragraph):
            parag_words = TextNormalizer.tokenize_words(parag_sent)
            wmd = w2v.wmdistance(quest_words, parag_words)
            if not np.isinf(wmd):
                dist_sum += wmd
                dist_denom += 1.0
                min_dist = min( min_dist, wmd )
                max_dist = max( max_dist, wmd )

        df.loc[index, 'min_wmd'] = min_dist
        df.loc[index, 'max_wmd'] = max_dist
        df.loc[index, 'avg_wmd'] = dist_sum/dist_denom


dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

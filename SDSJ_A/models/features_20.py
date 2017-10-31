# -*- coding: utf-8 -*-
# coding: utf-8
"""
Расширение набора фич - посимвольные расстояния и похожести.
Запускать после features_19.py.
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
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import scipy.spatial.distance
import gc
from nltk.metrics.distance import jaccard_distance, edit_distance
import distance
from fuzzywuzzy import fuzz
from nltk.stem.snowball import RussianStemmer
from Segmenter import Segmenter
from Tokenizer import Tokenizer

# --------------------------------------------------------------------------------------

stopwords = set(pd.read_csv("../data/stopwords.txt", encoding='utf-8')['word'].values)
stemmer = RussianStemmer()

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# --------------------------------------------------------------------------------------

word2lemma = dict()


def nonstop(words):
    return filter(lambda z: z not in stopwords, words)


def normalize_word(word):
    nword = word.lower().replace(u'ё', u'е')
    if nword in word2lemma:
        nword = word2lemma[nword]
    if len(nword) > 4:
        #return nword[:4]
        return stemmer.stem(nword)
    else:
        return nword

#tokenizer_regex = re.compile(r'[%s\s]+' % re.escape(u'[; .,?!-…№”“\'"–—_:«»*]()'))


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s


def normalize_word2(w):
    return w.lower().replace(u'ё', u'е')

def normalize_sent(s):
    return s.lower().replace(u'ё', u'е')

tokenizer = Tokenizer()
#def tokenize_stems(phrase):
#    return [normalize_word(w) for w in tokenizer_regex.split(phrase) if len(w) > 0]
def stemmize(phrase):
    return u' '.join( [normalize_word(w) for w in tokenizer.tokenize(phrase) if len(w) > 0] )

# -------------------------------------------------------------------------

with codecs.open('../data/word2lemma.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 3:
            word = tx[0].lower()
            lemma = tx[1].lower()
            word2lemma[word] = lemma

# --------------------------------------------------------------------------

segmenter = Segmenter()

def get_shingles3( s ):
    return [ c1+c2+c3 for (c1,c2,c3) in zip(s,s[1:],s[2:]) ]

# ----------------------------------------------------------------

print('Segmentation of paragraph texts...')

for name, df in [('train', dftrain), ('test', dftest)]:

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Calculating charwise similarities for " + name):

        question = row.question
        paragraph = row.paragraph

        quest_stems = stemmize(question)
        quest_shingles3 = get_shingles3(quest_stems)

        max_shingles3 = 0.0
        max_fuzz_qratio = 0.0
        max_fuzz_WRatio = 0.0
        max_fuzz_partial_ratio = 0.0
        max_fuzz_partial_token_set_ratio = 0.0
        max_fuzz_partial_token_sort_ratio = 0.0
        max_fuzz_token_set_ratio = 0.0
        max_fuzz_token_sort_ratio = 0.0

        for parag_sent in segmenter.split(paragraph):
            parag_stems = stemmize(parag_sent)

            # chars_dist = edit_distance( normalize_word2(quest), normalize_word2(parag), substitution_cost=1, transpositions=True)
            # min_chars_edit_dist = min( min_chars_edit_dist, chars_dist )
            shingles3 = distance.jaccard( quest_shingles3, get_shingles3(parag_stems))

            fuzz_qratio = 0.01 * fuzz.QRatio(quest_stems, parag_stems)
            max_fuzz_qratio = max( max_fuzz_qratio, fuzz_qratio )

            fuzz_WRatio = 0.01 * fuzz.WRatio(quest_stems, parag_stems)
            max_fuzz_WRatio = max( max_fuzz_WRatio, fuzz_WRatio )

            fuzz_partial_ratio = 0.01 * fuzz.partial_ratio(quest_stems, parag_stems)
            max_fuzz_partial_ratio = max( max_fuzz_partial_ratio, fuzz_partial_ratio )

            fuzz_partial_token_set_ratio = 0.01 * fuzz.partial_token_set_ratio(quest_stems, parag_stems)
            max_fuzz_partial_token_set_ratio = max( max_fuzz_partial_token_set_ratio, fuzz_partial_token_set_ratio)

            fuzz_partial_token_sort_ratio = 0.01 * fuzz.partial_token_sort_ratio(quest_stems, parag_stems)
            max_fuzz_partial_token_sort_ratio = max( max_fuzz_partial_token_sort_ratio, fuzz_partial_token_sort_ratio )

            fuzz_token_set_ratio = 0.01 * fuzz.token_set_ratio(quest_stems, parag_stems)
            max_fuzz_token_set_ratio = max( max_fuzz_token_set_ratio, fuzz_token_set_ratio)

            fuzz_token_sort_ratio = 0.01 * fuzz.token_sort_ratio(quest_stems, parag_stems)
            max_fuzz_token_sort_ratio = max( max_fuzz_token_sort_ratio, fuzz_token_sort_ratio )

        df.loc[index, 'max_shingles3_str'] = max_shingles3
        df.loc[index, 'max_fuzz_qratio_str'] = max_fuzz_qratio
        df.loc[index, 'max_fuzz_WRatio_str'] = max_fuzz_WRatio
        df.loc[index, 'max_fuzz_partial_ratio_str'] = max_fuzz_partial_ratio
        df.loc[index, 'max_fuzz_partial_token_set_ratio_str'] = max_fuzz_partial_token_set_ratio
        df.loc[index, 'max_fuzz_partial_token_sort_ratio_str'] = max_fuzz_partial_token_sort_ratio
        df.loc[index, 'max_fuzz_token_set_ratio_str'] = max_fuzz_token_set_ratio
        df.loc[index, 'max_fuzz_token_sort_ratio_str'] = max_fuzz_token_sort_ratio

# ----------------------------------------------------------------------------

print('Storing datasets...')
dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')






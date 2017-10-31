# -*- coding: utf-8 -*-
# coding: utf-8
"""
Добавочные фичи по named entries
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import scipy.spatial.distance
import gc
from nltk.stem.snowball import RussianStemmer
from Segmenter import Segmenter
from Tokenizer import Tokenizer
import TextNormalizer
import StopWords

# -------------------------------------------------------------------

funcwords = set()
funcword_px = [u'наречие', u'союз', u'частица', u'междометие', u'инфинитив', u'глагол', u'местоим_сущ', u'безлич_глаг']
with codecs.open('../data/word2lemma.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx)==3:
            word = tx[0].lower()
            p_o_s = tx[2].lower()
            if p_o_s in funcword_px:
                funcwords.add(word)

stopwords = StopWords.get_stopwords()

def nonstop(words):
    return filter(lambda z: z not in stopwords, words)


def is_NE(word):
    nword = TextNormalizer.normalize_word(word)
    return len(word)>1 and nword not in funcwords\
           and nonstop( nword ) and word[0].isupper()

def filter_NEs(tokens):
    return [ TextNormalizer.crop_word(word) for word in filter( is_NE, tokens ) ]

def NEs_intersection(words1, words2):
    return len( set(filter_NEs(words1)) & set(filter_NEs(words2)) )

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# ------------------------------------------------------------------------

segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing named entries features for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_raw(question)
        df.loc[index, 'quest_ne_count'] = len(filter_NEs(quest_words))

        parag_words = TextNormalizer.tokenize_raw(paragraph)
        df.loc[index, 'nes_intersection'] = NEs_intersection( quest_words, parag_words )

        max_ne_match = 0
        for parag_sent in segmenter.split(paragraph):
            parag_words = TextNormalizer.tokenize_raw(parag_sent)
            ne_match = NEs_intersection( quest_words, parag_words )
            max_ne_match = max( max_ne_match, ne_match )

        df.loc[index, 'best_ne_match'] = max_ne_match


dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

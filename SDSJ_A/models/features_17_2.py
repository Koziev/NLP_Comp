# -*- coding: utf-8 -*-
# coding: utf-8
"""
Добавочные фичи по иноязычным токенам
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


def is_cyrword(word):
    # cyrillic 0x0400...0x04FF
    return 0x0400 <= ord(word[0]) <= 0x04ff

def is_digit(word):
    return word[0].isdigit()


# is foreign language word
def is_FW(word):
    nword = TextNormalizer.normalize_word(word)
    return len(nword)>0 and not (is_cyrword(nword) or is_digit(nword))

def filter_FWs(tokens):
    return [ TextNormalizer.crop_word(word) for word in filter( is_FW, tokens ) ]

def FWs_intersection(words1, words2):
    return len( set(filter_FWs(words1)) & set(filter_FWs(words2)) )

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
        df.loc[index, 'quest_fw_count'] = len(filter_FWs(quest_words))

        parag_words = TextNormalizer.tokenize_raw(paragraph)
        df.loc[index, 'fw_intersection'] = FWs_intersection( quest_words, parag_words )

        max_ne_match = 0
        for parag_sent in segmenter.split(paragraph):
            parag_words = TextNormalizer.tokenize_raw(parag_sent)
            ne_match = FWs_intersection( quest_words, parag_words )
            max_ne_match = max( max_ne_match, ne_match )

        df.loc[index, 'best_fw_match'] = max_ne_match


dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

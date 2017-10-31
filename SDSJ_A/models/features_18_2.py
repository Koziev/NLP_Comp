# -*- coding: utf-8 -*-
# coding: utf-8
"""
Расширение набора фич - посимвольные расстояния и похожести.
Запускать после features_18_1.py.
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
import TextNormalizer

# --------------------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# --------------------------------------------------------------------------------------

def stemmize(phrase):
    return u' '.join( TextNormalizer.tokenize_stems(phrase) )

# -------------------------------------------------------------------------

segmenter = Segmenter()

print('Segmentation of paragraph texts...')

for name, df in [('train', dftrain), ('test', dftest)]:

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Calculating charwise similarities for " + name):

        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_stems = stemmize(question)

        max_fuzz_partial_ratio = 0.0

        for parag_sent in segmenter.split(paragraph):
            parag_stems = stemmize(parag_sent)
            fuzz_partial_ratio = 0.01 * fuzz.partial_ratio(quest_stems, parag_stems)
            max_fuzz_partial_ratio = max( max_fuzz_partial_ratio, fuzz_partial_ratio )

        df.loc[index, 'max_fuzz_partial_ratio_str'] = max_fuzz_partial_ratio

# ----------------------------------------------------------------------------

print('Storing datasets...')
dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

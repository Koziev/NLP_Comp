# -*- coding: utf-8 -*-
# coding: utf-8
"""
Похожесть вопроса на одно из предложений параграфа.
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
import Abbrev

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

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Calculating question similarities for " + name):

        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_stems = TextNormalizer.tokenize_stems( u' '.join(TextNormalizer.preprocess_question(question)) )
        quest_set = set(quest_stems)
        denom = float(len( quest_set ))

        max_intersect3 = 0.0

        for parag_sent in segmenter.split(paragraph):
            parag_stems = TextNormalizer.tokenize_stems( Abbrev.normalize_abbrev(parag_sent) )
            intersect3 = len( quest_set & set(parag_stems) ) / denom
            max_intersect3 = max( max_intersect3, intersect3 )

        df.loc[index, 'max_intersect3'] = max_intersect3

# ----------------------------------------------------------------------------

print('Storing datasets...')
dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

# -*- coding: utf-8 -*-

# Похожесть хвоста вопроса на предложения в параграфе

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

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# ------------------------------------------------------------------------

segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Computing the tail matching for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_crops(question)[4:]

        max_tail_match = 0
        for parag_sent in segmenter.split(paragraph):
            parag_words = TextNormalizer.tokenize_crops(parag_sent)
            tail_match = len( set(quest_words) & set(parag_words) )
            max_tail_match = max( max_tail_match, tail_match )

        df.loc[index, 'max_tail_match'] = max_tail_match

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

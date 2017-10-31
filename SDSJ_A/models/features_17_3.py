# -*- coding: utf-8 -*-

# Расстояние Кульбака — Лейблера https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5_%D0%9A%D1%83%D0%BB%D1%8C%D0%B1%D0%B0%D0%BA%D0%B0_%E2%80%94_%D0%9B%D0%B5%D0%B9%D0%B1%D0%BB%D0%B5%D1%80%D0%B0

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

target0_words = Counter()
target1_words = Counter()
df = dftrain[['question','target']]

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    quest_words = TextNormalizer.tokenize_raw(TextNormalizer.preprocess_question_str(row.question))
    if row['target']==0:
        target0_words.update(quest_words)
    else:
        target1_words.update(quest_words)


total0 = sum(target0_words.values())
word_0_freq = dict([ (w,f/total0) for (w,f) in target0_words.iteritems()])

total1 = sum(target1_words.values())
word_1_freq = dict([ (w,f/total1) for (w,f) in target1_words.iteritems()])

# ------------------------------------------------------------------------

segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    sum_w0 = []
    sum_w1 = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing named entries features for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        quest_words = TextNormalizer.tokenize_raw(row.question)
        sum_w0.append( sum( [math.pow(word_0_freq[w],2) for w in quest_words if w in word_0_freq] ) )
        sum_w1.append( sum( [math.pow(word_1_freq[w],2) for w in quest_words if w in word_1_freq] ) )

    df['sum_w0'] = sum_w0
    df['sum_w1'] = sum_w1

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

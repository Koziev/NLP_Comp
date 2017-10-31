# -*- coding: utf-8 -*-

# Количество несловарных слов в вопросе, и похожесть набора несловарных слов
# в вопросе и в предложении параграфа.

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
import codecs

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

known_words = set()

# русский лексикон
with codecs.open('../data/word2lemma.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 3:
            word = tx[0].lower()
            known_words.add(word)

# ------------------------------------------------------------------------

segmenter = Segmenter()

def is_oov(word):
    return TextNormalizer.normalize_word(word) not in known_words and len(word)>0 and not word[0].isdigit()


for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Computing the tail matching for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        quest_words = TextNormalizer.tokenize_words(question)
        quest_oov_words = set( map( TextNormalizer.crop_word, filter( is_oov, quest_words ) ) )
        df.loc[index, 'quest_oov_count'] = len(quest_oov_words)

        parag_words = TextNormalizer.tokenize_words(paragraph)
        parag_oov_words = set( map( TextNormalizer.crop_word, filter( is_oov, parag_words ) ) )
        df.loc[index, 'parag_oov_count'] = len(parag_oov_words)

        oov_intersection = len( quest_oov_words & parag_oov_words )
        df.loc[index, 'oov_intersection'] = oov_intersection

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

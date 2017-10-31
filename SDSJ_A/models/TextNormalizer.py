# -*- coding: utf-8 -*-
# coding: utf-8
"""
Функции для нормализации текста - уборка мусора, замена сокращений и т.д.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
import collections
from collections import Counter
import tqdm
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import functools
import sys
import codecs
from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from Tokenizer import Tokenizer
import Abbrev
from pymystem3 import Mystem


# -----------------------------------------------------------
ru_stemmer = RussianStemmer()
en_stemmer = EnglishStemmer()

m = Mystem()

tokenizer = Tokenizer()

word2lemma = dict()

# русский лексикон
with codecs.open('../data/word2lemma.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 3:
            word = tx[0].lower()
            lemma = tx[1].lower()
            word2lemma[word] = lemma

# английский лексикон
with codecs.open('../data/english_lemmas.csv', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx)==2:
            word = tx[0].lower().replace(u' - ', u'-')
            if word[0]==u'"' and word[-1]==u'"' and len(word)>2:
                word = word[1:-2]
            lemma = tx[1].lower().replace(u' - ', u'-')

            if word != lemma:
                if word in word2lemma:
                    if len(word2lemma[word])>len(lemma): # сохраняем самую короткую лемму
                        word2lemma[word] = lemma
                else:
                    word2lemma[word] = lemma

# -----------------------------------------------------------

def preprocess_line(text0):
    text = re.sub(u'\\[[0-9]+\\]', u'', text0)
    return text

# -----------------------------------------------------------

def split_word(word):
    if len(word)>1 and word[-1]=='%':
        return [ word[:-1], word[-1:] ]
    else:
        return [word]


def normalize_word(word):
    nword = word.lower().replace(u'ё', u'е').replace(u'\u0301', u'').replace(u'­', u'')
    return nword


def lemmatize_word(word):
    nword = normalize_word(word)
    if nword in word2lemma:
        return word2lemma[nword]
    else:
        return nword


def crop_word(word):
    nword = lemmatize_word(word)
    if len(nword) > 4:
        return nword[:4]
    else:
        return nword


cyr_chars = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

def stem_word(word):
    nword = normalize_word(word)
    if len(nword)>0:
        if nword[0] in cyr_chars:
            return ru_stemmer.stem(nword)
        else:
            return en_stemmer.stem(nword)
    else:
        return nword;


# -----------------------------------------------------------

def tokenize_raw(phrase):
    res_list = []
    for token in tokenizer.tokenize_raw(phrase):
        for word in split_word(token):
            if len(word)>0:
                res_list.append( word )

    return res_list

# -----------------------------------------------------------

def tokenize_words(phrase):
    res_list = []
    for token in tokenizer.tokenize(phrase):
        for word in split_word(token):
            if len(word)>0:
                res_list.append( normalize_word(word) )

    return res_list

# -----------------------------------------------------------

def tokenize_lemmas(phrase):
    #res_list = []
    #for token in tokenizer.tokenize(phrase):
    #    for word in split_word(token):
    #        if len(word)>0:
    #            res_list.append( lemmatize_word(word) )
    #
    #return res_list
    s2 = u' '.join(tokenizer.tokenize(phrase))
    return [l for l in m.lemmatize(s2) if len(l.strip())>0 ]

# -----------------------------------------------------------

def tokenize_stems(phrase):
    res_list = []
    for token in tokenizer.tokenize(phrase):
        for word in split_word(token):
            if len(word)>0:
                res_list.append( stem_word(word) )

    return res_list

# -----------------------------------------------------------

def tokenize_crops(phrase):
    res_list = []

    for token in m.lemmatize(phrase):
        if len(token)>0:
            for word in split_word(token):
                if len(word)>0:
                    res_list.append( word[:4] )

    return res_list

# ---------------------------------------------------------------

q_words = set(u'чей куда откуда что чему чем чего кто кого кем кому ком '
              u'чьего чьих чьими чьей почему чьи '
              u'как где когда сколько каков какова каковы какой какая какое какие '
              u'какого какой каких каким какими какому каким каком зачем какую '.split())

def preprocess_question(text0):
    text = Abbrev.normalize_abbrev(text0.strip())
    text = preprocess_line(text)
    words = [ word for word in tokenize_raw( text ) if normalize_word(word) not in q_words ]
    return words


def preprocess_question_str(text0):
    return u' '.join(preprocess_question(text0))

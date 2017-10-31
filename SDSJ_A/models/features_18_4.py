# -*- coding: utf-8 -*-



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

syn_stems = set()

with codecs.open('../data/word2similar.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 2:
            word1 = tx[0].lower()
            word2 = tx[1].lower()

            stem1 = TextNormalizer.stem_word(word1)
            stem2 = TextNormalizer.stem_word(word2)
            if stem1!=stem2:
                syn_stems.add( (stem1, stem2) )
                syn_stems.add( (stem2, stem1) )

# ------------------------------------------------------------------------

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# ------------------------------------------------------------------------

print('Loading the w2v model...')
# w2v = gensim.models.KeyedVectors.load_word2vec_format(r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt', binary=False)
w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/eek/polygon/w2v/w2v.CBOW=1_WIN=20_DIM=32.txt',
                                                      binary=False)
vec_len = len(w2v.syn0[0])
print('Vector length={0}'.format(vec_len))

# ------------------------------------------------------------------------


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s

# ------------------------------------------------------------------------

def calc_similarity( quest, parag ):
    quest_words = set(TextNormalizer.tokenize_words(quest))

    parag_words = set(TextNormalizer.tokenize_words(parag))
    parag_lemmas = set(TextNormalizer.tokenize_lemmas(parag))
    parag_stems = set(TextNormalizer.tokenize_stems(parag))
    parag_crops = set(TextNormalizer.tokenize_crops(parag))

    matched_parag_words = set()

    sim = 0.0
    for qword in quest_words:
        if qword in parag_words:
            matched_parag_words.add(qword)
            sim += 1.0
        else:
            qlemma = TextNormalizer.lemmatize_word(qword)
            if qlemma in parag_lemmas:
                #matched_parag_lemmas.add(qlemma)
                sim += 1.0
            else:
                qstem = TextNormalizer.stem_word(qword)
                if qstem in parag_stems:
                    sim += 0.95
                else:
                    qcrop = TextNormalizer.crop_word(qword)
                    if qcrop in parag_crops:
                        sim += 0.80
                    else:
                        found_syn = False
                        for pstem in parag_stems:
                            if (qstem,pstem) in syn_stems:
                                sim += 0.70
                                found_syn = True
                                break

                        if not found_syn:
                            if qword in w2v:
                                qvec = w2v[qword]
                                max_cos = -1e38
                                for pword in parag_words:
                                    if pword in w2v:
                                        pvec = w2v[pword]
                                        c = v_cosine( qvec, pvec )
                                        max_cos = max( max_cos, c )

                                sim += max_cos*0.5

    return sim / len(quest_words)




# ------------------------------------------------------------------------


segmenter = Segmenter()

for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Synonyms matching for " + name):
        question = TextNormalizer.preprocess_question_str(row.question)
        paragraph = row.paragraph

        max_sim = 0.0
        for parag_sent in segmenter.split(paragraph):
            sim = calc_similarity(question, parag_sent)
            max_sim = max( max_sim, sim )

        df.loc[index, 'max_syn_sim'] = max_sim

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

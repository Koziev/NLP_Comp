# -*- coding: utf-8 -*-
# coding: utf-8
"""
Расширение набора фич - LSA похожесть для каждого предложения параграфа vs вопрос.
Запускать после features_17.py.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import scipy.spatial.distance
import gc
import TextNormalizer
import StopWords
from Segmenter import Segmenter

# --------------------------------------------------------------------------------------

stopwords = StopWords.get_stopwords()

dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

# --------------------------------------------------------------------------------------

def nonstop(words):
    return filter(lambda z: z not in stopwords, words)


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s

# --------------------------------------------------------------------------

segmenter = Segmenter()

# ----------------------------------------------------------------

print('Segmentation of paragraph texts and preparing TF-IDF corpus.')

tfidf_corpus = set()
train_rowid_2_i = dict()
train_quest_crops = []
train_parag_indeces = []
train_parag_texts = []
test_parag_texts = []
test_parag_indeces = []
test_quest_crops = []
test_rowid_2_i = dict()

for name, df in [('train', dftrain), ('test', dftest)]:

    quest_crops = []
    parag_texts = []
    parag_indeces = []
    rowid_2_i = dict()

    row_i = 0
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Segmentation for " + name):

        rowid_2_i[index] = row_i

        parag_indeces.append(index)
        question = row.question
        paragraph = row.paragraph

        question_crops = TextNormalizer.tokenize_crops(question)

        tfidf_corpus.add( u' '.join(question_crops) )
        quest_crops.append( question_crops )

        parag_sents = []
        for parag_sent in segmenter.split(paragraph):
            parag_crops = TextNormalizer.tokenize_crops(parag_sent)
            if len(parag_crops)>0:
                tfidf_corpus.add( u' '.join(parag_crops) )
                parag_sents.append(parag_crops)

        parag_texts.append(parag_sents)
        row_i += 1

    if name=='train':
        train_parag_indeces = parag_indeces
        train_quest_crops = quest_crops
        train_parag_texts = parag_texts
        train_rowid_2_i = rowid_2_i
    else:
        test_parag_indeces = parag_indeces
        test_quest_crops = quest_crops
        test_parag_texts = parag_texts
        test_rowid_2_i = rowid_2_i

# ----------------------------------------------------------------------------

# ========================== LSA ==========================

print('Performing LSA with {} documents'.format(len(tfidf_corpus)))

vectorizer = TfidfVectorizer(max_features=None, ngram_range=(1, 1), min_df=1, analyzer='word', stop_words=stopwords)

svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=50, random_state=123456)

svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
svd_transformer.fit(tfidf_corpus)

del tfidf_corpus
gc.collect()

# теперь выполняем трансформацию предложений в параграфах и вопросах в LSA-пространство
for name, df in [('train', dftrain), ('test', dftest)]:

    # векторизация вопросов
    lsa_questions = None
    if name=='train':
        lsa_questions = train_quest_crops
    else:
        lsa_questions = test_quest_crops

    lsa_questions = [ (u' '.join(crops)) for crops in lsa_questions ]

    print('LSA tranform for {} questions in {}...'.format(len(lsa_questions), name))
    questions_lsv = svd_transformer.transform(lsa_questions)

    # векторизация предложений в параграфах
    rowid_2_sentids = dict()
    parags = None
    parag_indeces = None
    lsa_parags = []
    rowid_2_i = None
    if name=='train':
        parags = train_parag_texts
        parag_indeces = train_parag_indeces
        rowid_2_i = train_rowid_2_i
    else:
        parags = test_parag_texts
        parag_indeces = test_parag_indeces
        rowid_2_i = test_rowid_2_i

    total_sents = sum( len(p) for p in parags )

    running_j = 0
    for i, (parag_index, parag_crops) in enumerate( zip(parag_indeces, parags) ):
        ix = []
        for sent in parag_crops:
            ix.append(running_j)
            parag_sent_str = u' '.join(sent)
            lsa_parags.append(parag_sent_str)
            running_j += 1

        rowid_2_sentids[parag_index] = ix

    print('LSA tranform for {} paragraph lines in {}...'.format(len(lsa_parags), name))
    parags_lsv = svd_transformer.transform(lsa_parags)

    # теперь можем добавлять новую фичу - максимальная похожесть среди предложений параграфа и вопросом.
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="computing LSA features for " + name):
        i = rowid_2_i[index]
        question_v = questions_lsv[i] # LSA-вектор для вопроса

        # теперь найдем максимальную похожесть этого вопроса и предложений в параграфе
        max_lsa = -1e38
        min_lsa = 1e38
        avg_lsa_num = 0.0

        parag_ix = rowid_2_sentids[index] # линейные индексы записей в parags_lsv для предложений в обрабатываемой записи

        for i in parag_ix:
            parag_sent_v = parags_lsv[i]

            lsa_sim = v_cosine(question_v, parag_sent_v)
            max_lsa = max( lsa_sim, max_lsa )
            min_lsa = min( lsa_sim, min_lsa )
            avg_lsa_num += lsa_sim

        df.loc[index, 'max_lsa_per_sent'] = max_lsa
        df.loc[index, 'min_lsa_per_sent'] = min_lsa
        df.loc[index, 'avg_lsa_per_sent'] = avg_lsa_num/len(parag_ix)

# --------------------------------------------------------------------------


print('Storing datasets...')
dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

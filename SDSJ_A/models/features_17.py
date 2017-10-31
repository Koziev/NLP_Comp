# -*- coding: utf-8 -*-
# coding: utf-8
"""
Подготовка основного набора фич для конкурса https://contest.sdsj.ru/
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
import collections
from nltk.stem.snowball import RussianStemmer
from Segmenter import Segmenter
from Tokenizer import Tokenizer
import TextNormalizer

stopwords = set(pd.read_csv("../data/stopwords.txt", encoding='utf-8')['word'].values)

LSA_DIMS = 100
add_las_vectors = True


#dftrain = pd.read_csv("../input/train_task1_latest.csv", encoding='utf-8')
#dftest = pd.read_csv("../input/sdsj_A_test.csv", encoding='utf-8')
dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')

word2lemma = dict()


def nonstop(words):
    return filter(lambda z: z not in stopwords, words)


def normalize_word(word):
    nword = word.lower().replace(u'ё', u'е')
    if nword in word2lemma:
        nword = word2lemma[nword]
    if len(nword) > 4:
        return nword[:4]
    else:
        return nword


#tokenizer_regex = re.compile(r'[%s\s]+' % re.escape(u'[; .,?!-…№”“\'"–—_:«»*]()'))


def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s


def normalize_word2(w):
    return w.lower().replace(u'ё', u'е')


tokenizer = Tokenizer()
def tokenize2(phrase):
    return [normalize_word2(w) for w in tokenizer.tokenize(phrase) if len(w) > 0]


def uniq_words(text):
    return set( TextNormalizer.tokenize_crops(text) )


def uniq_words2(text):
    return set(nonstop(TextNormalizer.tokenize_crops(text)))


def calculate_idfs(data):
    counter_paragraph = Counter()
    uniq_paragraphs = data['paragraph'].unique()
    for paragraph in tqdm.tqdm(uniq_paragraphs, desc="calc idf"):
        set_words = uniq_words(paragraph)
        counter_paragraph.update(set_words)

    num_docs = uniq_paragraphs.shape[0]
    idfs = {}
    for word in counter_paragraph:
        idfs[word] = np.log(num_docs / counter_paragraph[word])
    return idfs



with codecs.open('../data/word2lemma.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 3:
            word = tx[0].lower()
            lemma = tx[1].lower()
            word2lemma[word] = lemma


# -------------------------------------------------------------------------

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


# --------------------------------------------------------------------------

def tokenize3(phrase):
    return TextNormalizer.tokenize_raw(phrase)

def is_NE(word):
    nword = TextNormalizer.normalize_word(word)
    return len(word)>1 and nword not in funcwords\
           and nonstop( nword ) and word[0].isupper()

def filter_NEs(tokens):
    return [ TextNormalizer.crop_word(word) for word in filter( is_NE, tokens ) ]

def NEs_intersection(words1, words2):
    return len( set(filter_NEs(words1)) & set(filter_NEs(words2)) )

# --------------------------------------------------------------------------


segmenter = Segmenter()

def tokenize4(s):
    return TextNormalizer.tokenize_crops(s)


def best_match( parag, quest, idfs ):
    max_sent_match = 0.0
    max_ne_match=0.0

    max_idf_intersection2_sent = 0.0

    q_words4_list = tokenize4(quest)
    q_words4 = set(q_words4_list)
    q_denom = len(q_words4)
    q_words3 = set(tokenize3(quest))

    question2 = uniq_words2(quest)

    for parag_sent in segmenter.split(parag):
        ps_words4_list = tokenize4(parag_sent)
        ps_words4 = set(ps_words4_list)
        match = len(q_words4&ps_words4) / q_denom
        max_sent_match = max( max_sent_match, match )

        ps_words3 = set(filter_NEs(tokenize3(parag_sent)))
        match = len(q_words3&ps_words3)
        max_ne_match = max( max_ne_match, match )

        paragraph2 = uniq_words2(parag_sent)
        idf_intersection2 = np.sum([idfs.get(x, 0.0) for x in paragraph2 & question2])
        max_idf_intersection2_sent = max( max_idf_intersection2_sent, idf_intersection2)

    return (max_sent_match, max_ne_match, max_idf_intersection2_sent)


# ========================== LSA ==========================
tfidf_corpus = set()
for name, df in [('train', dftrain), ('test', dftest)]:
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="extracting texts for LSA from " + name):
        question = TextNormalizer.tokenize_crops( TextNormalizer.preprocess_question_str(row.question) )
        paragraph = TextNormalizer.tokenize_crops( row.paragraph )
        tfidf_corpus.add(u' '.join(question))
        tfidf_corpus.add(u' '.join(paragraph))


vectorizer = TfidfVectorizer(max_features=None, ngram_range=(1, 1), min_df=1, analyzer='word', stop_words=stopwords)

svd_model = TruncatedSVD(n_components=LSA_DIMS, algorithm='randomized', n_iter=20, random_state=42)

svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
svd_transformer.fit(tfidf_corpus)

del tfidf_corpus
gc.collect()

# -----------------------------------------------------------------------------------

idfs = calculate_idfs(dftrain)

# -----------------------------------------------------------------------------------


for name, df in [('train', dftrain), ('test', dftest)]:

    lsa_questions = []
    lsa_paragraphs = []

    questions = []
    paragraphs = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="build features for " + name):
        quest_str = TextNormalizer.preprocess_question_str(row.question)

        question = uniq_words(quest_str)
        paragraph = uniq_words(row.paragraph)
        df.loc[index, 'len_paragraph'] = len(paragraph)
        df.loc[index, 'len_question'] = len(question)
        df.loc[index, 'len_intersection'] = len(paragraph & question)
        df.loc[index, 'idf_question'] = np.sum([idfs.get(word, 0.0) for word in question])
        df.loc[index, 'idf_paragraph'] = np.sum([idfs.get(word, 0.0) for word in paragraph])
        df.loc[index, 'idf_intersection'] = np.sum([idfs.get(word, 0.0) for word in paragraph & question])

        question2 = uniq_words2(quest_str)
        paragraph2 = uniq_words2(row.paragraph)
        df.loc[index, 'len_paragraph2'] = len(paragraph2)
        df.loc[index, 'len_question2'] = len(question2)
        df.loc[index, 'len_intersection2'] = len(paragraph2 & question2)
        df.loc[index, 'idf_question2'] = np.sum([idfs.get(word, 0.0) for word in question2])
        df.loc[index, 'idf_paragraph2'] = np.sum([idfs.get(word, 0.0) for word in paragraph2])
        df.loc[index, 'idf_intersection2'] = np.sum([idfs.get(word, 0.0) for word in paragraph2 & question2])

        question_crops = TextNormalizer.tokenize_crops(quest_str)
        paragraph_crops = TextNormalizer.tokenize_crops(row.paragraph)

        questions.append(question_crops)
        paragraphs.append(paragraph_crops)

        lsa_questions.append(u' '.join(question_crops))
        lsa_paragraphs.append(u' '.join(paragraph_crops))

        df.loc[index, 'nes_intersection'] = NEs_intersection( tokenize3(quest_str), tokenize3(row.paragraph) )

        (max_sent_match, max_ne_match, max_idf_intersection2_sent) = best_match(row.paragraph, quest_str, idfs)
        df.loc[index, 'best_sent_match'] = max_sent_match
        df.loc[index, 'best_ne_match'] = max_ne_match
        df.loc[index, 'max_idf_intersection2_sent'] = max_idf_intersection2_sent


    print('LSA tranform for questions in {}...'.format(name))
    questions_ls = svd_transformer.transform(lsa_questions)
    print('LSA tranform for paragraphs in {}...'.format(name))
    paragraphs_ls = svd_transformer.transform(lsa_paragraphs)

    print('Calculating the LSA cos simirarity in {}...'.format(name))
    cos_dist = np.zeros((len(questions_ls), 1))
    for i in range(len(questions_ls)):
        cos_dist[i, 0] = v_cosine(paragraphs_ls[i], questions_ls[i])

    df['lsa_cos'] = cos_dist

    if add_las_vectors:
        nrow = len(questions_ls)
        ncol = LSA_DIMS
        quest_column_data = np.zeros(nrow)
        parag_column_data = np.zeros(nrow)

        for icol in tqdm.tqdm(range(ncol), total=ncol, desc="Storing LSA vectors for " + name):
            for jrow in range(nrow):
                quest_column_data[jrow] = questions_ls[jrow][icol]
                parag_column_data[jrow] = paragraphs_ls[jrow][icol]

            df['lsa_quest_{}'.format(icol)] = quest_column_data
            df['lsa_parag_{}'.format(icol)] = parag_column_data

dftrain.to_csv("../data/dftrain.csv", index=True, encoding='utf-8')
dftest.to_csv("../data/dftest.csv", index=True, encoding='utf-8')

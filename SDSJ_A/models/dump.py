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
import codecs


# ------------------------------------------------------------------------

df = pd.read_csv("../data/dftrain17.csv", encoding='utf-8')

segmenter = Segmenter()

#test_str = u'Сразу после возвращения Фрама Нансен стал главным специалистом по полярным исследованиям в мире, по выражению Р. Хантфорда — оракулом для всех исследователей полярных широт Севера и Юга [188]. Нансен консультировал бельгийского барона Адриена де Жерлаша, который планировал в 1898 году свою экспедицию в Антарктиду, одним из участников команды был Руаль Амундсен[189]. Известнейший исследователь Гренландии Кнуд Расмуссен сравнил посещение Нансена с посвящением в рыцари[190]. В то же время Нансен категорически отказался встречаться со своим соотечественником Карстеном Борхгревинком, сочтя его мошенником, хотя именно он совершил первую успешную зимовку на побережье Антарктиды[191]. В 1900 году в Норвегию приехал за консультациями Роберт Скотт со своим покровителем Клементом Маркхэмом — давним другом Нансена, готовившим британскую экспедицию в Антарктиду. Несмотря на то, что англичане практически проигнорировали все советы, Нансен и Скотт остались в хороших отношениях[192].'
#for l in segmenter.split(test_str):
#    print(l)

# -------------------------------------------------------------------------

wrt0 = codecs.open('../data/rows(y=0).txt', 'w', 'utf-8')
wrt1 = codecs.open('../data/rows(y=1).txt', 'w', 'utf-8')

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0] ):
    question = row.question
    paragraph = row.paragraph

    wrt = wrt0 if row['target']==0 else wrt1

    question1 = TextNormalizer.preprocess_question_str(question)
    quest_stems = set(TextNormalizer.tokenize_stems( question1 ))
    quest_crops = set(TextNormalizer.tokenize_crops( question1 ))

    denom_stems = float(len(quest_stems))
    denom_crops = float(len(quest_crops))

    max_crop_match = 0.0
    best_crop_match_sent = u''

    max_stem_match = 0.0
    best_stem_match_sent = u''

    wrt.write('\n\nid={}\n'.format(index))
    for i,parag_sent in enumerate(segmenter.split(paragraph)):
        wrt.write(u'P[{}]\t{}\n'.format(i, parag_sent))

        parag_stems = set(TextNormalizer.tokenize_stems( parag_sent ))
        parag_crops = set(TextNormalizer.tokenize_crops( parag_sent ))

        match_stems = len(parag_stems&quest_stems)/denom_stems
        match_crops = len(parag_crops&quest_crops)/denom_crops

        if match_stems>max_stem_match:
            max_stem_match = match_stems
            best_stem_match_sent = parag_sent

        if match_crops>max_crop_match:
            max_crop_match = match_crops
            best_crop_match_sent = parag_sent

    wrt.write(u'Q\t{}\n'.format(question))

    q2 = u' '.join(TextNormalizer.preprocess_question(question))
    wrt.write(u'Z\t{}\n'.format(q2))

    wrt.write(u'Stem match ==> {} with {}\n'.format(max_stem_match, best_stem_match_sent))
    wrt.write(u'Crop match ==> {} with {}\n'.format(max_crop_match, best_crop_match_sent))

wrt0.close()
wrt1.close()



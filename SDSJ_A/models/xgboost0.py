# -*- coding: utf-8 -*-
'''
Решение задачи конкурса https://contest.sdsj.ru/dashboard?problem=A 
(c) Козиев Илья inkoziev@gmail.com

Справка по XGBoost:
http://xgboost.readthedocs.io/en/latest/
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
import xgboost
import codecs
import sklearn.metrics
import TrainTestSplitter
import TextNormalizer
import Segmenter


dftrain = pd.read_csv("../data/dftrain.csv", encoding='utf-8')
dftest = pd.read_csv("../data/dftest.csv", encoding='utf-8')


# ----------------------------------------------------------------------------------
columns = ['lsa_cos', 'w2v_cos',
           'len_paragraph',
           'len_question',
           'len_intersection',
           'idf_question',
           'idf_paragraph',
           'idf_intersection',
           'len_paragraph2',
           'len_question2',
           'len_intersection2',
           'idf_question2', 'idf_paragraph2', 'idf_intersection2',
           'best_sent_match',
           'max_w2v_cos',
           'avg_w2v_cos',
           'max_idf_intersection2_sent',
           'quest_ne_count',
           'nes_intersection',
           #'best_ne_match',
           #'quest_fw_count',
           #'fw_intersection',
           #'best_fw_match',
           'sum_w0',
           'sum_w1',
           'max_tail_match',
           'max_lsa_per_sent',
           'min_lsa_per_sent',
           'avg_lsa_per_sent',
           #'quest_oov_count',
           'parag_oov_count', 'oov_intersection', # features_18_1.py
           'max_intersect3',  # features_18_3.py
           'max_syn_sim',  # features_18_4.py
           #'rnn_cos'
           'min_wmd',
           'max_wmd',
           'avg_wmd',
           #'wmd'
           'max_fuzz_partial_ratio_str',
           ] # cv.test.logloss_mean=0.0543836 public=0.9316628479258103

#columns.extend( ['quest_rnn_{}'.format(i) for i in range(32)] )
#columns.extend( ['parag_rnn_{}'.format(i) for i in range(32)] )

#LSA_DIMS = 100
#columns.extend( ['lsa_quest_{}'.format(i) for i in range(LSA_DIMS)] )
#columns.extend( ['lsa_parag_{}'.format(i) for i in range(LSA_DIMS)] )


max_depth = 5
colsample_bytree = 0.90
colsample_bylevel = 0.90
# ----------------------------------------------------------------------------------


X_train = dftrain[columns]
X_submit = dftest[columns]
y_train = dftrain['target']

# ------------------------------------------------------------------------

D_train = xgboost.DMatrix(X_train, y_train, silent=0)
D_submit = xgboost.DMatrix(X_submit, silent=0)

xgb_params = {
    'booster': 'gbtree',  # 'dart' | 'gbtree',
    # 'n_estimators': _n_estimators,
    'subsample': 1.0,
    'max_depth': max_depth,
    'seed': 123456,
    'min_child_weight': 1,
    'eta': 0.12,  # 0.12,
    'gamma': 0.01,
    'colsample_bytree': colsample_bytree,
    'colsample_bylevel': colsample_bylevel,
    'scale_pos_weight': 1.0,
    'eval_metric': 'logloss',  # 'auc' | 'logloss',
    'objective': 'binary:logistic',
    'silent': 1,
    # 'updater': 'grow_gpu'
}

print('Compute best number of estimators using xgboost.cv')
cvres = xgboost.cv(xgb_params,
                   D_train,
                   # booster="gbtree",
                   num_boost_round=10000,
                   nfold=5,
                   early_stopping_rounds=50,
                   metrics=['logloss'],  # metrics=['logloss','auc'],
                   seed=123456,
                   # callbacks=[eta_cb],
                   verbose_eval=50,
                   # verbose=True,
                   # print_every_n=50,
                   )

cvres.to_csv('../submit/cvres.csv')
nbrounds = cvres.shape[0]
print('CV finished, nbrounds={}'.format(nbrounds))

cv_logloss = cvres['test-logloss-mean'].tolist()[-1]
cv_std = cvres['test-logloss-std'].tolist()[-1]
print('cv.test.logloss_mean={}'.format(cv_logloss))
print('cv.test.logloss_std={}'.format(cv_std))

print('Train model...')
cl = xgboost.train(xgb_params, D_train,
                   num_boost_round=nbrounds,
                   verbose_eval=False,
                   # callbacks=[eta_cb]
                   )
print('Training is finished')

feature_scores = cl.get_fscore()
with open('../submit/feature_xgboost_scores.txt', 'w') as wrt:
    for (feature, score) in sorted(feature_scores.iteritems(), key=lambda z: -z[1]):
        wrt.write('{}\t{}\n'.format(feature, score))

print('Compute submission...')
y_submission = cl.predict(D_submit, ntree_limit=nbrounds)
dftest['prediction'] = y_submission

dftest[['paragraph_id', 'question_id', 'prediction']].to_csv("../submit/submit18.csv", index=False)


# ------------------------------------------------------

print('Validation')
X_train, X_test, y_train, y_test, i_train, i_test = TrainTestSplitter.split_val(dftrain, columns)
D_train = xgboost.DMatrix(X_train, y_train, silent=0)
D_test = xgboost.DMatrix(X_test, y_test, silent=0)

print('Train model for validation...')
cl = xgboost.train(xgb_params, D_train,
                   num_boost_round=nbrounds,
                   verbose_eval=False,
                   )

print('Validating...')
y_pred = cl.predict(D_test, ntree_limit=nbrounds)
val_score = sklearn.metrics.roc_auc_score(y_test, y_pred)
print('roc_auc_score={}'.format(val_score))

y_2 = [ (1 if z>0.5 else 0) for z in y_pred ]
acc = sklearn.metrics.accuracy_score( y_test, y_2 )
print('accuracy_score={}'.format(acc) )


segmenter = Segmenter.Segmenter()

if False:
    print('Printing mispredictons')
    y_test = y_test.values
    with codecs.open('../submit/mispredictions.txt', 'w', 'utf-8') as wrt:
        for y_i,df_i in enumerate(i_test):
            if y_2[y_i]!=y_test[y_i]:
                wrt.write('\n\ny_pred={} y_test={}\n'.format(y_pred[y_i], y_test[y_i]))
                quest = dftrain.loc[ df_i, ['question'] ].values[0]
                parag = dftrain.loc[ df_i, ['paragraph'] ].values[0]
                quest_str = TextNormalizer.preprocess_question_str(quest)

                for j,parag_sent in enumerate(segmenter.split(parag)):
                    wrt.write(u'P[{}]:\t{}\n'.format(j, parag_sent))

                wrt.write(u'Q:\t{}\n'.format(quest_str))


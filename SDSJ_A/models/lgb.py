# -*- coding: utf-8 -*-
# coding: utf-8

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
import codecs
import sklearn.metrics
import TrainTestSplitter
import TextNormalizer
import Segmenter
import lightgbm


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
           #'rnn_cos',
           'min_wmd',
           'max_wmd',
           'avg_wmd',
           'wmd',
           'max_fuzz_partial_ratio_str',
           ] # cv.test.logloss_mean= public=

#columns.extend( ['quest_rnn_{}'.format(i) for i in range(32)] )
#columns.extend( ['parag_rnn_{}'.format(i) for i in range(32)] )

#columns.extend( ['lsa_quest_{}'.format(i) for i in range(32)] )
#columns.extend( ['lsa_parag_{}'.format(i) for i in range(32)] )

# ----------------------------------------------------------------------------------


X_train = dftrain[columns]
X_submit = dftest[columns]
y_train = dftrain['target']


D_train = lightgbm.Dataset( X_train, label=y_train )
D_submit = lightgbm.Dataset( X_submit )

# --------------------------------------------------------------

def get_params():
    px = dict()

    px['boosting_type']='gbdt'
    px['objective'] ='binary'
    px['metric'] = 'binary_logloss'
    px['learning_rate'] = 0.02
    px['num_leaves'] = 40
    px['min_data_in_leaf'] = 40
    px['min_sum_hessian_in_leaf'] = 1
    px['max_depth'] = -1
    px['lambda_l1'] = 0.0  # space['lambda_l1'],
    px['lambda_l2'] = 0.0  # space['lambda_l2'],
    px['max_bin'] = 256
    px['feature_fraction'] = 0.95
    px['bagging_fraction'] = 0.95
    px['bagging_freq'] = 5

    return px

# --------------------------------------------------------------

lgb_params = get_params()

cvres = lightgbm.cv(lgb_params,
                    D_train,
                    num_boost_round=10000,
                    nfold=5,
                    #metrics='binary_logloss',
                    stratified=False,
                    shuffle=True,
                    #fobj=None,
                    #feval=None,
                    #init_model=None,
                    #feature_name='auto',
                    #categorical_feature='auto',
                    early_stopping_rounds=50,
                    #fpreproc=None,
                    verbose_eval=False,
                    show_stdv=False,
                    seed=123456,
                    #callbacks=None
                    )

nbrounds = len(cvres['binary_logloss-mean'])
cv_logloss = cvres['binary_logloss-mean'][-1]

print( 'CV finished nbrounds={} loss={:7.5f}'.format( nbrounds, cv_logloss ) )

cl = lightgbm.train(lgb_params,
                    D_train,
                    num_boost_round=nbrounds,
                    # metrics='mlogloss',
                    # valid_sets=None,
                    # valid_names=None,
                    # fobj=None,
                    # feval=None,
                    # init_model=None,
                    # feature_name='auto',
                    # categorical_feature='auto',
                    # early_stopping_rounds=None,
                    # evals_result=None,
                    verbose_eval=False,
                    # learning_rates=None,
                    # keep_training_booster=False,
                    # callbacks=None
                    )


# ------------------------------------------------------------------------
print('Compute submission...')
y_submission = cl.predict(X_submit, num_iteration=nbrounds)
dftest['prediction'] = y_submission
dftest[['paragraph_id', 'question_id', 'prediction']].to_csv("../submit/lgb_submit.csv", index=False)



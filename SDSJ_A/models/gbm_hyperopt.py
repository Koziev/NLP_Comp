# -*- coding: utf-8 -*-
'''
Решение задачи конкурса https://contest.sdsj.ru/dashboard?problem=A 
(c) Козиев Илья inkoziev@gmail.com

Поиск оптимальных значений для sklearn GradientBoostingClassifier с помощью гипероптимизации
библиотекой hyperopt.

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''

from __future__ import print_function
from __future__ import division

import pandas as pd
import xgboost
import codecs
import numpy as np

import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import time
import colorama  # https://pypi.python.org/pypi/colorama
from uuid import uuid4
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import sklearn.metrics
import TrainTestSplitter
import TextNormalizer
import Segmenter


# кол-во случайных наборов гиперпараметров
N_HYPEROPT_PROBES = 5000

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest


submission_folder = '../submit'

# --------------------------------------------------------------------------------

colorama.init()

# --------------------------------------------------------------------------------

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
           ] # cv.test.logloss_mean=0.0543836 public=0.9316628479258103

# ----------------------------------------------------------------------------------


X_train = dftrain[columns]
X_submit = dftest[columns]
y_train = dftrain['target']

# ---------------------------------------------------------------------

def random_str():
    return str(uuid4())

# ---------------------------------------------------------------------

obj_call_count = 0
max_score = -1e38

log_writer = open( os.path.join(submission_folder, 'gbm-hyperopt-log.txt'), 'w' )

def objective(space):
    global obj_call_count, max_score

    start = time.time()

    obj_call_count += 1

    print('\nGBM objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,max_score) )

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))

    cl = GradientBoostingClassifier(loss='deviance',
                                    learning_rate=space['learning_rate'],
                                    n_estimators=int(space['n_estimators']),
                                    subsample=space['subsample'],
                                    criterion='friedman_mse',
                                    min_samples_split=int(space['min_samples_split']),
                                    min_samples_leaf=int(space['min_samples_leaf']),
                                    min_weight_fraction_leaf=0.0,
                                    max_depth=int(space['max_depth']),
                                    #min_impurity_decrease=1e-07,
                                    init=None,
                                    random_state=123456,
                                    max_features=None,
                                    verbose=0,
                                    max_leaf_nodes=None, warm_start=False, presort='auto')

    cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=123456)
    cv_score = np.mean( cross_val_score( cl, X_train, y=y_train, scoring='roc_auc', cv=cv ) )
    print('cv_score={}'.format(cv_score))

    do_submit = False
    if cv_score>max_score:
        max_score = cv_score
        do_submit = True
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(max_score) + colorama.Fore.RESET)

    if do_submit:
        submit_guid = random_str()

        print('Compute submission guid={}'.format(submit_guid))

        submission_filename = os.path.join(submission_folder,
                                           'gbm_score={:13.11f}_submission_guid={}.csv'.format(cv_score, submit_guid))

        print('Train model...')
        cl.fit(X_train,y_train)
        print('Training is finished')
        y_submission = cl.predict_proba(X_submit)[:, 1]
        dftest['prediction'] = y_submission

        dftest[['paragraph_id', 'question_id', 'prediction']].to_csv(submission_filename, index=False)

        log_writer.write( 'cv_score={:<7.5f} Params:{} submit_guid={}\n'.format( cv_score, str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]), submit_guid ) )
        log_writer.flush()


    end = time.time()
    elapsed = int(end - start)
    #print('elapsed={}'.format(elapsed ) )

    return{'loss':-cv_score, 'status': STATUS_OK }




# --------------------------------------------------------------------------------

space ={
        'max_depth': hp.quniform("max_depth", 4, 6, 1),
        'n_estimators': hp.quniform("n_estimators", 200, 1500, 1),
        #'min_child_weight': hp.quniform ('min_child_weight', 0, 10, 1),
        'subsample': hp.uniform ('subsample', 0.75, 1.0),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -3, -1.6),
       }

# --------------------------------------------------------------


trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

log_writer.close()

print('-'*50)
print('The best params:')
print( best )
print('\n\n')

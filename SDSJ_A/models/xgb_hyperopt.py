# -*- coding: utf-8 -*-
'''
Решение задачи конкурса https://contest.sdsj.ru/dashboard?problem=A 
(c) Козиев Илья inkoziev@gmail.com

Справка по XGBoost:
http://xgboost.readthedocs.io/en/latest/

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''

from __future__ import print_function
from __future__ import division

import pandas as pd
import xgboost
import numpy as np
import os
import time
from uuid import uuid4

import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama  # https://pypi.python.org/pypi/colorama


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

# ------------------------------------------------------------------------

D_train = xgboost.DMatrix(X_train, y_train, silent=0)
D_submit = xgboost.DMatrix(X_submit, silent=0)


# ---------------------------------------------------------------------

def random_str():
    return str(uuid4())

# ---------------------------------------------------------------------


def get_xgboost_params(space):
    _max_depth = int(space['max_depth'])
    _min_child_weight = space['min_child_weight']
    _subsample = space['subsample']
    _gamma = space['gamma'] if 'gamma' in space else 0.01
    _eta = space['eta']
    _seed = 123456
    _colsample_bytree = space['colsample_bytree']
    _colsample_bylevel = space['colsample_bylevel']
    booster = 'gbtree'

    #sorted_params = sorted(space.iteritems(), key=lambda z: z[0])

    xgb_params = {
        'booster': booster,
        'subsample': _subsample,
        'max_depth': _max_depth,
        'seed': _seed,
        'min_child_weight': _min_child_weight,
        'eta': _eta,
        'gamma': _gamma,
        'colsample_bytree': _colsample_bytree,
        'colsample_bylevel': _colsample_bylevel,
        'scale_pos_weight': 1.0,
        'eval_metric': 'logloss', #'auc',  # 'logloss',
        'objective': 'binary:logistic',
        'silent': 1,
    }

    return xgb_params


# -----------------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf

log_writer = open( os.path.join(submission_folder, 'xgb-hyperopt-log.txt'), 'w' )

def objective(space):
    global obj_call_count, cur_best_loss

    start = time.time()

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    xgb_params = get_xgboost_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]))

    cvres = xgboost.cv(xgb_params,
                       D_train,
                       num_boost_round=10000,
                       nfold=7,
                       early_stopping_rounds=50,
                       metrics=['logloss'],  # metrics=['logloss','auc'],
                       seed=123456,
                       # callbacks=[eta_cb],
                       verbose_eval=50,
                       # verbose=True,
                       # print_every_n=50,
                       )

    #cvres.to_csv('../submit/cvres.csv')
    nbrounds = cvres.shape[0]
    print('CV finished, nbrounds={}'.format(nbrounds))

    cv_logloss = cvres['test-logloss-mean'].tolist()[-1]
    cv_std = cvres['test-logloss-std'].tolist()[-1]
    print('cv.test.logloss_mean={}'.format(cv_logloss))
    #print('cv.test.logloss_std={}'.format(cv_std))

    do_submit = cv_logloss<cur_best_loss
    if cv_logloss<cur_best_loss:
        cur_best_loss = cv_logloss
        do_submit = True
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    if do_submit:
        submit_guid = random_str()

        print('Compute submission guid={}'.format(submit_guid))

        submission_filename = os.path.join(submission_folder,
                                           'xgboost_loss={:13.11f}_submission_guid={}.csv'.format(cv_logloss, submit_guid))

        print('Train model...')
        cl = xgboost.train(xgb_params, D_train,
                           num_boost_round=nbrounds,
                           verbose_eval=False,
                           # callbacks=[eta_cb]
                           )
        print('Training is finished')

        y_submission = cl.predict(D_submit, ntree_limit=nbrounds)
        dftest['prediction'] = y_submission

        dftest[['paragraph_id', 'question_id', 'prediction']].to_csv(submission_filename, index=False)

        log_writer.write( 'holdout_logloss={:<7.5f} Params:{} nbrounds={} submit_guid={}\n'.format( cv_logloss, str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]), nbrounds, submit_guid ) )
        log_writer.flush()


    end = time.time()
    elapsed = int(end - start)
    #print('elapsed={}'.format(elapsed ) )

    return{'loss':cv_logloss, 'status': STATUS_OK }




# --------------------------------------------------------------------------------

space ={
        'max_depth': hp.quniform("max_depth", 5, 6, 1),
        'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1),
        'subsample': hp.uniform ('subsample', 0.75, 1.0),
        'gamma': hp.loguniform('gamma', -5.0, 0.0),
        'eta': hp.loguniform('eta', -3, -1.6),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.90, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.90, 1.0),
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



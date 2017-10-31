# -*- coding: utf-8 -*-
'''
Решение задачи конкурса https://contest.sdsj.ru/dashboard?problem=A 
(c) Козиев Илья inkoziev@gmail.com


LightGBM:
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#lightgbm-package

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
'''

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import os
import time
from uuid import uuid4

import lightgbm
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
           #'rnn_cos',
           'min_wmd',
           'max_wmd',
           'avg_wmd',
           'wmd',
           'max_fuzz_partial_ratio_str',
           ] # cv.test.logloss_mean= public=

# ----------------------------------------------------------------------------------


X_train = dftrain[columns]
X_submit = dftest[columns]
y_train = dftrain['target']

D_train = lightgbm.Dataset( X_train, label=y_train )

# ---------------------------------------------------------------------

def random_str():
    return str(uuid4())

# ---------------------------------------------------------------------


def get_params(space):
    px = dict()

    px['boosting_type']='gbdt'
    px['objective'] ='binary'
    px['metric'] = 'binary_logloss'
    px['learning_rate']=space['learning_rate']
    px['num_leaves'] = int(space['num_leaves'])
    px['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    px['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    px['max_depth'] = int(space['max_depth']) if 'max_depth' in space else -1
    px['lambda_l1'] = 0.0  # space['lambda_l1'],
    px['lambda_l2'] = 0.0  # space['lambda_l2'],
    px['max_bin'] = 256
    px['feature_fraction'] = space['feature_fraction']
    px['bagging_fraction'] = space['bagging_fraction']
    px['bagging_freq'] = 5

    return px


# -----------------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf

log_writer = open( os.path.join(submission_folder, 'lgb-hyperopt-log.txt'), 'w' )

def objective(space):
    global obj_call_count, cur_best_loss

    start = time.time()

    obj_call_count += 1

    print('\nLightGBM objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    lgb_params = get_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params]))

    cvres = lightgbm.cv(lgb_params,
                        D_train,
                        num_boost_round=10000,
                        nfold=5,
                        # metrics='binary_logloss',
                        stratified=False,
                        shuffle=True,
                        # fobj=None,
                        # feval=None,
                        # init_model=None,
                        # feature_name='auto',
                        # categorical_feature='auto',
                        early_stopping_rounds=50,
                        # fpreproc=None,
                        verbose_eval=False,
                        show_stdv=False,
                        seed=123456,
                        # callbacks=None
                        )

    nbrounds = len(cvres['binary_logloss-mean'])
    cv_logloss = cvres['binary_logloss-mean'][-1]
    print('CV finished nbrounds={} loss={:7.5f}'.format(nbrounds, cv_logloss))

    do_submit = cv_logloss<cur_best_loss
    if cv_logloss<cur_best_loss:
        cur_best_loss = cv_logloss
        do_submit = True
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    if do_submit:
        submit_guid = random_str()

        print('Compute submission guid={}'.format(submit_guid))

        submission_filename = os.path.join(submission_folder,
                                           'lgb_loss={:13.11f}_submission_guid={}.csv'.format(cv_logloss, submit_guid))

        print('Train model...')
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
        print('Training is finished')

        y_submission = cl.predict(X_submit, num_iteration=nbrounds)
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
        'num_leaves': hp.quniform ('num_leaves', 20, 100, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -3, -1.5),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
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



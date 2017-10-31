# -*- coding: utf-8 -*-
# coding: utf-8
"""
Подготовка основного набора фич для конкурса https://contest.sdsj.ru/
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
from sklearn.model_selection import train_test_split

def split_val( dftrain, columns ):
    X = dftrain[columns]
    y = dftrain['target']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.50, random_state=1234562)
    i_train, i_test, _, _ = train_test_split( range(0,dftrain.shape[0]), y, test_size=0.50, random_state=1234562)
    return (X_train, X_test, y_train, y_test, i_train, i_test)

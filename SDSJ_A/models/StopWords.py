# -*- coding: utf-8 -*-
# coding: utf-8
"""
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

from nltk.corpus import stopwords
import pandas as pd

stopwords = set(pd.read_csv("../data/stopwords.txt", encoding='utf-8')['word'].values) | set(stopwords.words('english'))

def get_stopwords():
    return stopwords
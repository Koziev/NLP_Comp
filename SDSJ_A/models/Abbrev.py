# -*- coding: utf-8 -*-
# coding: utf-8

def normalize_abbrev(text0):
    text = text0

    sx = [(u'т.п.', u'т_п_'), (u'т. п.', u'т_п_'), (u'т.д.', u'т_д_'), (u'т. д.', u'т_д_'),
          (u'др.', u'др_'), (u'л.с.', u'л_с_'), (u'л. с.', u'л_с_'),
          (u'н.э.', u'н_э_'), (u'н. э.', u'н_э_'), (u'пр.', u'пр_'), (u'сокр.', u'сокр_'),
          (u'мн.', u'мн_'), (u'разг.', u'разг_'), (u'проф.', u'проф_'), (u'греч.', u'греч_'),
          (u'п.', u'п_'), (u'англ.', u'англ_'), (u'совр.', u'совр_'), (u'г.', u'г_'),
          (u'Св.', u'Св_'), (u'св.', u'св_'), (u' в.', u' в_'), (u'русск.', u'русск_'),
          (u'т.н.', u'т_н_'), (u'т. н.', u'т_н_'), (u'итал.', u'итал_'), (u'лат.', u'лат_')]

    for s1, s2 in sx:
        text = text.replace(s1, s2)

    return text


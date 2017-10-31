# Определение релевантности вопроса к тексту.

Решения задачи A в конкурсе https://contest.sdsj.ru (оценка 0.93773 на закрытой части).

Скрипты features_*.py генерируют столбцы с фичами в .../input/dftrain.csv и dftest.csv

Остальные скрипты - разные решатели:

nn5.py - сеточная модель, со сверточными и рекуррентными фрагментами.

gbm_hyperopt.py - на базе sklearn:GradientBoostingClassifier плюс hyperopt

lgb.py - на базе LightGBM

xgboost0.py - на базе XGBoost

lgb_hyperopt.py

xgb_hyperopt.py


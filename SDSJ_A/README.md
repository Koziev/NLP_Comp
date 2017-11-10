# Определение релевантности вопроса к тексту.

Решения [задачи A](https://github.com/sberbank-ai/data-science-journey-2017) в конкурсе https://contest.sdsj.ru (оценка 0.93773 на закрытой части).

Скрипты features_*.py генерируют столбцы с фичами в .../input/dftrain.csv и dftest.csv

Остальные скрипты - разные классификаторы:

nn5.py - сеточная модель, со сверточными и рекуррентными фрагментами, на базе [Keras](https://keras.io/)  
gbm_hyperopt.py - решение на базе [sklearn:GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) плюс [hyperopt](https://github.com/hyperopt/hyperopt)  
lgb.py и lgb_hyperopt.py - решение на базе LightGBM  
xgboost0.py и xgb_hyperopt.py - решение на базе XGBoost  



#! /opt/anaconda3/bin/python

from hpsklearn import HyperoptEstimator, random_forest, any_classifier
from sklearn.externals import joblib
from imblearn.pipeline import Pipeline
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import geopy
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84
import dill

data = pd.read_csv("data/Kaggle_YourCabs_training.csv")
X, y = data.drop('Car_Cancellation', axis=1), data['Car_Cancellation']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

feature_pipe = dill.load(open('feature_pipe.pkl', 'rb'))

#feature_pipe.fit(X_train, y_train)
#X_train_f = feature_pipe.transform(X_train).as_matrix()
#X_test_f = feature_pipe.transform(X_test).as_matrix()

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from hyperopt import tpe

#rf = random_forest()
clf = RandomForestClassifier(n_estimators=100)
svc = SVC()
bbc = BalancedBaggingClassifier(base_estimator=clf,
                                n_estimators=1000, n_jobs=10)

estim = HyperoptEstimator(classifier=any_classifier('my_clf', n_jobs=5),
                          max_evals=200,
                          preprocessing=[], trial_timeout=120, algo=tpe.suggest,
                          verbose=True)

smote = [('smote', SMOTE())]
ru = [('ru', RandomUnderSampler())]
clf_pipe = [('clf', estim)]
final_pipe_list = feature_pipe.steps + smote + clf_pipe
final_pipe = Pipeline(final_pipe_list)


final_pipe.fit(X_train, y_train.as_matrix())

dill.dump(final_pipe, open('any_smote_kwargs_neu.pkl', 'wb'))

#print(.score(X_test, y_test.as_matrix()))

from sklearn.metrics import classification_report
y_pred = final_pipe.predict(X_test)
print(classification_report(y_test, y_pred))

hopt = final_pipe.steps[-1][1]
bm = hopt.best_model()['learner']


best_pipe_list = feature_pipe.steps + smote + [('bm', bm)]
best_pipe.fit(X_train, y_train.as_matrix())
dill.dump(final_pipe, open('best_bm_smote_any.pkl', 'wb'))

print("End")

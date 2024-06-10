from turtle import color
import pandas as pd
import shap
import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn import metrics
from titanic import get_titanic_data
from ensamble import get_ensamble


X_train, y_train, X_test, y_test, clf = get_ensamble()
X_train, X_test, y_train, y_test, train, test = get_titanic_data('train', 'features')

model = clf
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, prediction))

explainer = shap.Explainer(model.predict, test)
shap_values = explainer(train)
# # print(shap_values.size)
sample_ind = 19

shap.plots.waterfall(shap_values[sample_ind])
shap.plots.beeswarm(shap_values.abs, color="shap_red")
shap.plots.beeswarm(shap_values, max_display=14)
shap.plots.bar(shap_values)
shap.plots.heatmap(shap_values) 
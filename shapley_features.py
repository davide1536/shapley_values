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


# a classic housing price dataset
# X,y = shap.datasets.boston()
# X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
# cancer = load_breast_cancer()
# print(cancer.feature_names)
# data = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
# print(data.describe())
# print(data.shape)

# data = data.drop(['worst radius', 'worst area', 'mean area', 'worst perimeter'], axis = 1)

# print(data.shape)

# # x = data.iloc[:, 0:29]
# # y = cancer.target
# # print("y ", y.shape)
# # print("x ", x.shape)
# # x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.33, random_state=42)
# data["target"] = cancer.target
# train, test = train_test_split (data, test_size=0.33, random_state=42)
# X_train = train.iloc[:, 0:26]
# print(X_train)
# X_test = test.iloc[:, 0:26]
# y_train = train.iloc[:,-1]
# y_test = test.iloc[:,-1]

# train = train.drop(['target'],  axis = 1)
# test = test.drop(['target'], axis = 1)

# print("X_train ", X_train.shape)
# print("y_train ", y_train.shape)
# print("X_test ", X_test.shape)
# print("y_test ", y_test.shape)
X_train, y_train, X_test, y_test, clf = get_ensamble()
X_train, X_test, y_train, y_test, train, test = get_titanic_data('train', 'features')

# print("X:", X)
# print("Y: ", y)
# X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.33, random_state=42)
# # a simple linear model
#model = sklearn.linear_model.LinearRegression()
# y = train.iloc
model = clf
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, prediction))
# print("Model coefficients:\n")
# for i in range(data.shape[1]):
#     print(data.colu)
explainer = shap.Explainer(model.predict, test)
shap_values = explainer(train)
# # print(shap_values.size)
sample_ind = 19
# shap.partial_dependence_plot(
#     "Pclass", model.predict, test, model_expected_value=True,
#     feature_expected_value=True, ice=False,
#     shap_values=shap_values[sample_ind:sample_ind+1,:]
# )
shap.plots.waterfall(shap_values[sample_ind])
shap.plots.beeswarm(shap_values.abs, color="shap_red")
shap.plots.beeswarm(shap_values, max_display=14)
shap.plots.bar(shap_values)
shap.plots.heatmap(shap_values) 
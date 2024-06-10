from cProfile import label
import enum
from cv2 import solve
import numpy as np
from shapley import PermutationSampler
from ensamble import retrieve_weights, get_ensamble, ensamble_neural_network
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from collections import defaultdict
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt


'''
In this class I compute the actual shapley values for ensamble models. I also used methods to generate experimental graphs.

'''

def compute_shap(weights):
    W = np.array(weights)
    

    W = W/W.sum()
    q = 0.5
    solver = PermutationSampler()

    print("quote ", q)
    solver.solve_game(W,q)
    shapley_values = solver.get_solution()
    return shapley_values, solver

def target(mean_prediction):
    for i in range(len(mean_prediction)):
        if mean_prediction[i]>=0.5:
            mean_prediction[i] = 1
        else:
            mean_prediction[i] = 0
    return mean_prediction

def building_ensamble(trees, X_test, y_test):
    print("building")
    scores = []
    for i in range(1, len(trees)):
        mean_proba = np.zeros(len(X_test))
        proba = np.zeros(len(X_test))
        for j in range(i):
            predicted_proba = trees[j].predict(X_test)
            proba =proba + predicted_proba
        mean = proba/i
        prediction = target(mean)
        print("score: ", metrics.accuracy_score(y_test, prediction))
        scores.append(metrics.accuracy_score(y_test, prediction))
    return scores

def get_trees(dictionary, sort_shap):
    lista = []
    for shap in sort_shap:
        lista.append(dictionary[shap][0])
    return lista

def build_dictionaryNN (clf, avg_shap):

    dictionary = defaultdict(list)
    for i, estimator in enumerate(clf.estimators_):
        dictionary[len(estimator.coefs_[1]).bit_length() -1 ].append(avg_shap[i])
    return dictionary

def build_dictionary(clf, avg_shap):
    dictionary = defaultdict(list)
    
    for i, estimator in enumerate(clf.estimators_):
        dictionary[avg_shap[i]].append(estimator)
    return dictionary


def displayModelInformation(avg_shapley, clf):
    dictionary = build_dictionary(clf, avg_shapley)
    sort_avg_shapley = np.sort(avg_shapley)[::-1]
    plt.plot(sort_avg_shapley)
    plt.show()

    trees = get_trees(dictionary, sort_avg_shapley)
    scores = building_ensamble(trees, X_test, y_test)
    plt.plot(scores)
    plt.ylabel("accuracy")
    plt.xlabel("n ensamble")
    ax = plt.gca()
    plt.show()
# W = np.array([[0.99, 0.01, 0.29843549, 0.15379455, 0.51862131, 0.34891333,
#   0.1075523  ,0.77481699 ,0.27726541 ,0.88820124 ,0.43517843 ,0.49584311,
#   0.97304657 ,0.54722007 ,0.87338943 ,0.37438674 ,0.15430086 ,0.90116497,
#   0.90280463 ,0.21883294 ,0.27128867 ,0.88007508 ,0.72142279 ,0.20474932,
#   0.58962011 ,0.18098523 ,0.10759176 ,0.19045006 ,0.77667651 ,0.50374377,
#   0.14314454 ,0.28396882 ,0.72998475 ,0.86541802 ,0.69480103 ,0.82359642,
#   0.34568706 ,0.59626615 ,0.45889262 ,0.47196615 ,0.18462066 ,0.27060618,
#   0.14513884 ,0.97889627 ,0.57690177 ,0.86666917 ,0.81444338 ,0.76350051,
#   0.84789274 ,0.97741937 ,0.68141679 ,0.48756238 ,0.57635841 ,0.69576718,
#   0.45225732 ,0.76294495 ,0.98842759 ,0.16288184 ,0.58601026 ,0.28860111,
#   0.30734203 ,0.64452474 ,0.56254204 ,0.65260114 ,0.63892126 ,0.13173215,
#   0.41435691 ,0.41736315 ,0.13721167 ,0.46936669]])


#Compute random forest shapley values
X_train, y_train, X_test, y_test, clf = get_ensamble()
X_test, shap_x, y_test, shap_y = train_test_split(X_test,y_test, test_size=0.5, random_state=42)
weights= retrieve_weights(shap_x, shap_y, clf, 'random_forest')

shapley, solver = compute_shap(weights)
avg_shapley = solver.get_average_shapley()

#For each ensamble model display its shapley value and its performance
displayModelInformation(avg_shapley, clf, X_test, y_test)

#neural network complexity
X_train, y_train, X_test, y_test, clf = ensamble_neural_network()
weights = retrieve_weights(X_test, y_test, clf, 'neural')
print("length test",len(X_test))
print("length ensamble",len(clf.estimators_))
print("weight 0 ", len(weights))
print("weight 1 ",len(weights[1]))
shapley, solver = compute_shap(weights)
avg_shapley = solver.get_average_shapley()
dictionary = build_dictionaryNN(clf, avg_shapley)
labels, data = dictionary.keys(), dictionary.values()

plt.boxplot(data)

plt.xticks(range(1, len(labels) + 1), labels)
plt.show()










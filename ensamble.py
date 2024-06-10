import random
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier
from titanic import get_titanic_data
#hidden_layer_sizes=(n_nodes[random.randint(0,len(n_nodes)-1)],)

from sklearn.linear_model import Perceptron
def retrieve_neural_network():
    clfs = []
    n_nodes = [1 , 2, 128]
    #n_nodes = [1, 4, 9]
    for i in range(0,500):
        clf = MLPClassifier(hidden_layer_sizes=(n_nodes[random.randint(0,len(n_nodes)-1)],),max_iter=500)
        clfs.append((str(i),clf))
    return clfs

def ensamble_neural_network():
    #data = load_breast_cancer()
    X_train, X_test, y_train, y_test = get_titanic_data('train', 'ensamble')
    print("dimensione dataset train: ", X_train.shape)
    print("dimensione dataset test: ", X_test.shape)

    #X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.50, random_state=42)
    classifiers = retrieve_neural_network()
    clf = VotingClassifier(estimators= classifiers,voting='soft')
    clf.fit(X_train, y_train)
    return X_train, y_train, X_test, y_test, clf

def get_ensamble():
    X_train, X_test, y_train, y_test = get_titanic_data('train', 'ensamble')
    print("dimensione dataset train: ", X_train.shape)
    print("dimensione dataset test: ", X_test.shape)
   
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    return X_train, y_train, X_test, y_test, clf

def retrieve_weights(X_test, y_test, clf, type_task):
   
    total_proba = []
    
    for estimator in clf.estimators_:
       
       
        proba = estimator.predict_proba(X_test)
        if type_task == 'neural':
          
            print("classi:",estimator.classes_)
        
            
        total_proba.append(proba)
       

    total_weights = []
    for i in range(len(X_test)):
        weights = []
        for estimator in range(len(total_proba)):
            if (y_test[i] == 0):
                weights.append(total_proba[estimator][i][0]/len(total_proba))
            else:
                weights.append(total_proba[estimator][i][1]/len(total_proba))

        total_weights.append(weights)

    return total_weights #total_weights_noise






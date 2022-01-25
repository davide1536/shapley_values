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
    #data = load_breast_cancer()
    #X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.50, random_state=42)
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    return X_train, y_train, X_test, y_test, clf

def retrieve_weights(X_test, y_test, clf, type_task):
   

    #for each data point retrieve the individual weight of each classifier
    # print(clf.predict_proba(X_test), y_test)

    # for i in range(len(X_test)):
        #print("sample ",i)
    #total proba: first dimension: estimator, second dimension: sample, third dimension: distribution probabilities over classes
    total_proba = []
    #total_proba_noise = []
    for estimator in clf.estimators_:
        #noise = np.random.normal(0,1)
        #proba_noise = (0.3* estimator.predict_proba(X_test) + 0.7*noise)
       
        proba = estimator.predict_proba(X_test)
        if type_task == 'neural':
            # print("estimator:", len(estimator.coefs_[1]).bit_length() -1)
            # print("proba: ",proba)
            print("classi:",estimator.classes_)
        
            #print( str(len(estimator.coefs_[1]).bit_length() -1) + ":" + str(proba[0:20]))
        total_proba.append(proba)
        #total_proba_noise.append(proba_noise)
        print(estimator)
    #individual weight for each predictor for each data point
    #total_weights: first dimension: sample, second dimension: estimator 
    total_weights = []
    #total_weights_noise = []
    for i in range(len(X_test)):
        weights = []
        #weights_noise = []
        for estimator in range(len(total_proba)):
            #weights_noise.append(total_proba_noise[estimator][i][y_test[i]]/len(total_proba[0]))
            # if type_task == 'neural':
            #     print("estimator:", len(clf.estimators_[estimator].coefs_[1]).bit_length() -1)
            #     print(total_proba[estimator][i][y_test[i]])
            if (y_test[i] == 0):
                weights.append(total_proba[estimator][i][0]/len(total_proba))
            else:
                weights.append(total_proba[estimator][i][1]/len(total_proba))

        total_weights.append(weights)
        #total_weights_noise.append(weights_noise)

    return total_weights #total_weights_noise






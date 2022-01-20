from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.linear_model import Perceptron
def get_ensamble():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.50, random_state=42)
    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X_train, y_train)
    return X_train, y_train, X_test, y_test, clf

def retrieve_weights(X_test, y_test, clf):
   

    #for each data point retrieve the individual weight of each classifier
    # print(clf.predict_proba(X_test), y_test)

    # for i in range(len(X_test)):
        #print("sample ",i)
    #total proba: first dimension: estimator, second dimension: sample, third dimension: distribution probabilities over classes
    total_proba = []
    for estimator in clf.estimators_:
        total_proba.append(estimator.predict_proba(X_test))
        print(estimator)
    #individual weight for each predictor for each data point
    #total_weights: first dimension: sample, second dimension: estimator 
    total_weights = []
    for i in range(len(X_test)):
        weights = []
        for estimator in range(len(total_proba)):
            weights.append(total_proba[estimator][i][y_test[i]]/len(total_proba[0]))
        total_weights.append(weights)

    return total_weights






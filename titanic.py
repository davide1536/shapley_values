
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
directory = 'titanic/'
def get_titanic_data(mode, task):
    if mode == 'train':
        directory = 'titanic/train.csv'
    else:
        directory = 'titanic/test.csv'

    print(directory)
    df = pd.read_csv(directory)
    cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(cols, axis=1)

    dummies = []
    cols = ['Pclass', 'Sex', 'Embarked']
    for col in cols:
        dummies.append(pd.get_dummies(df[col]))

    titanic_dummies = pd.concat(dummies, axis=1)
    df = pd.concat((df, titanic_dummies), axis=1)
    df = df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

    df['Age'] = df['Age'].interpolate()
    
    
    y = df['Survived'].values
    df = df.drop(['Survived'], axis=1)
    X = df.values
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    train.columns = train.columns.astype(str)
    test.columns = test.columns.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    if (task == 'ensamble'):
        return X_train, X_test, y_train, y_test
    else: 
        return X_train, X_test, y_train, y_test, train, test

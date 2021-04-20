import numpy as np
import feature_extraction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from flask import jsonify


def getResult(url):

    #Importing dataset
    data = pd.read_csv('dataset_v3.csv')

    #Seperating features and labels
    X = data.drop('Result', axis=1)
    y = data['Result']

    #Seperating training features, testing features, training labels & testing labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # clf = DecisionTreeClassifier()
    # clf = BernoulliNB()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    total = accuracy_score(y_test, y_pred)

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(1,-1)

    try:
        prediction = clf.predict(X_new)
        if prediction == -1:
            return "Terindikasi sebagai website phishing."
        else:
            return "Bukan website phishing."
    except:
        prediction = clf.predict(X_new)
        return "Terindikasi sebagai website phishing."

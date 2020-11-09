import numpy as np
import feature_extraction
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.naive_bayes import GaussianNB
from flask import jsonify


def getResult(url):

    #Importing dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")

    #Seperating features and labels
    X = data[: , :-1]
    y = data[: , -1]

    #Seperating training features, testing features, training labels & testing labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    total = score*100

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(1,-1)

    try:
        prediction = clf.predict(X_new)
        akurasi = clf.score(X_new)
        if prediction == -1:
            return f"Phishing Url with accuracy {total}, and prediction {prediction}"
        else:
            return f"Legitimate Url, with accuracy {total}, and prediction {prediction}"
    except:
        return f"Phishing Url, with accuracy {total}"

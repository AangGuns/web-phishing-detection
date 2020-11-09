import numpy as np
import feature_extraction
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.naive_bayes import BernoulliNB
from flask import jsonify


def getResult(url):

    #Importing dataset
    data = pd.read_csv('dataset_v3.csv')

    #Seperating features and labels
    X = data.drop('Result', axis=1)
    y = data['Result']

    #Seperating training features, testing features, training labels & testing labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Remove constant, quasi constant, duplicate features
    constant_filter = VarianceThreshold(threshold=0.01)
    constant_filter.fit(X_train)
    X_train_filter = constant_filter.transform(X_train)
    X_test_filter = constant_filter.transform(X_test)

    X_train_T = X_train_filter.T
    X_test_T = X_test_filter.T

    X_train_T = pd.DataFrame(X_train_T)
    X_test_T = pd.DataFrame(X_test_T)

    #Remove duplicate features
    X_train_T.duplicated().sum()

    duplicated_features = X_train_T.duplicated()

    features_to_keep = [not index for index in duplicated_features]

    X_train_unique = X_train_T[features_to_keep].T
    X_test_unique = X_test_T[features_to_keep].T

    # Calculate the MI
    mi = mutual_info_classif(X_train_unique, y_train)
    mi = pd.Series(mi)
    mi.index = X_train_unique.columns

    mi.sort_values(ascending=False, inplace = True)

    sel = SelectPercentile(mutual_info_classif, percentile=80).fit(X_train_unique, y_train)
    X_train_unique.columns[sel.get_support()]

    X_train_mi = sel.transform(X_train_unique)
    X_test_mi = sel.transform(X_test_unique)


    clf = BernoulliNB()
    clf.fit(X_train_mi, y_train)
    score = clf.score(X_test_mi, y_test)
    total = score*100
    print("|| ====================================================== ||")
    print("Akurasi Model klasifikasi dengan seleksi fitur:")
    print(total)

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)

    indices = X_train_unique.columns[sel.get_support()]
    X_new_fill = [i for j, i in enumerate(X_new) if j in indices]

    X_new_fill = np.array(X_new_fill).reshape(1, -1)

    try:
        prediction = clf.predict(X_new_fill)
        if prediction == -1:
            return "Terindikasi sebagai website phishing."
        else:
            print ("Setelah diterapkan fitur seleksi:")
            print (X_new_fill)
            return "Bukan website phishing."
    except:
        prediction = clf.predict(X_new)
        return "Terindikasi sebagai website phishing."

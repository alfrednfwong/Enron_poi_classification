#!/usr/bin/python
import sys
sys.path.append("../tools/")
import pickle
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
FINANCIAL_FEATURES = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees']
EMAIL_FEATURES = ['from_poi_to_this_person', 'to_messages', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

# Import data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame(data_dict).transpose()
df = df[features_list]
df = df.replace("NaN", np.nan)

# Correct dirty and missing data
df = df.drop("TOTAL", axis=0)
df.loc["BELFER ROBERT", "deferred_income"] = -102500
df.loc["BELFER ROBERT", "deferral_payments"] = np.nan
df.loc["BELFER ROBERT", "expenses"] = 3285
df.loc["BELFER ROBERT", "director_fees"] = 102500
df.loc["BELFER ROBERT", "total_payments"] = 3285
df.loc["BELFER ROBERT", "exercised_stock_options"] = np.nan
df.loc["BELFER ROBERT", "restricted_stock"] = 44093
df.loc["BELFER ROBERT", "restricted_stock_deferred"] = -44093
df.loc["BELFER ROBERT", "total_stock_value"] = np.nan
df.loc["BHATNAGAR SANJAY", "other"] = np.nan
df.loc["BHATNAGAR SANJAY", "expenses"] = 137864
df.loc["BHATNAGAR SANJAY", "director_fees"] = np.nan
df.loc["BHATNAGAR SANJAY", "total_payments"] = 137864
df.loc["BHATNAGAR SANJAY", "exercised_stock_options"] = 15456290
df.loc["BHATNAGAR SANJAY", "restricted_stock"] = 2604490
df.loc["BHATNAGAR SANJAY", "restricted_stock_deferred"] = -2604490
df.loc["BHATNAGAR SANJAY", "total_stock_value"] = 15456290
df[FINANCIAL_FEATURES] = df[FINANCIAL_FEATURES].fillna(0)
df[EMAIL_FEATURES] = df[EMAIL_FEATURES].fillna(df.mean())

# Create new features
df["from_poi_ratio"] = df.from_poi_to_this_person / df.to_messages
df["share_receipt_with_poi_ratio"] = df.shared_receipt_with_poi / df.to_messages
df["to_poi_ratio"] = df.from_this_person_to_poi / df.from_messages
df["salary_ratio"] = df.salary / (df.total_payments + df.total_stock_value)
df["bonus_ratio"] = df.bonus / (df.total_payments + df.total_stock_value)
df.bonus_ratio = df.bonus_ratio.fillna(0)
df.salary_ratio = df.salary_ratio.fillna(0)
features_list += ["from_poi_ratio", "share_receipt_with_poi_ratio", "to_poi_ratio", "salary_ratio", "bonus_ratio"]

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = df.copy(deep=True)
df_scaled[features_list[1:]] = scaler.fit_transform(df_scaled[features_list[1:]])
my_dataset = df_scaled.transpose().to_dict()
data_scaled = featureFormat(my_dataset, features_list, sort_keys=False)
labels_scaled, features_scaled = targetFeatureSplit(data_scaled)

# Feature selection
recur_base_estimator = ensemble.RandomForestClassifier(n_estimators=20, min_samples_split=5, random_state=5)
recursive = RFECV(estimator=recur_base_estimator, scoring="f1", cv=10)
recursive.fit(features_scaled, labels_scaled)
recursive_result = pd.DataFrame([df.columns[1:], recursive.ranking_, recursive.grid_scores_]).transpose()
recursive_result.columns = ["feature", "ranking", "score"]
print(recursive_result.sort_values(by=["ranking", "score"]))

# Change feature selection
features_list = ["poi", "bonus", "exercised_stock_options", "total_stock_value", "share_receipt_with_poi_ratio",
                 "to_poi_ratio"]
my_dataset = df_scaled.transpose().to_dict()
data_scaled = featureFormat(my_dataset, features_list, sort_keys=False)
labels_scaled, features_scaled = targetFeatureSplit(data_scaled)

# Train test split
features_train, features_test, labels_train, labels_test = (
    train_test_split(features_scaled, labels_scaled, test_size=0.3, random_state=42)
)

# KNN with grid search CV
parameters = {
    "n_neighbors": [3, 5, 10, 20],
    "weights": ["distance", "uniform"],
    "p": [1, 2, 3, 4, 100]
}
knnbase = KNeighborsClassifier()
clf = GridSearchCV(knnbase, parameters, scoring="f1")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
proba_output = clf.predict_proba(features_test)
probs = []
for elem in proba_output:
    probs.append(elem[1])
print("Features:")
print(features_list)
print("\nClassifier:")
print(clf)
print("\nBest estimator")
print(clf.best_estimator_)
print("\nClassification report:")
print(metrics.classification_report(labels_test, pred))
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels_test, pred))

# Random forest with grid search CV
parameters = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10, 15, 20]
}
forestbase = ensemble.RandomForestClassifier(n_estimators=10)
clf = GridSearchCV(forestbase, parameters, scoring="f1")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
proba_output = clf.predict_proba(features_test)
probs = []
for elem in proba_output:
    probs.append(elem[1])
print("Features:")
print(features_list)
print("\nClassifier:")
print(clf)
print("\nBest estimator")
print(clf.best_estimator_)
print("\nClassification report:")
print(metrics.classification_report(labels_test, pred))
print("\nConfusion matrix:")
print(metrics.confusion_matrix(labels_test, pred))

# adaboosted decision tree
for min_samples_split in [2, 5, 10, 15, 20, 30]:
    print("\n#########################")
    print("min_samples_split = {}\n".format(min_samples_split))
    clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=min_samples_split),
                                           learning_rate = 0.5, n_estimators= 100)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    proba_output = clf.predict_proba(features_test)
    probs = []
    for elem in proba_output:
        probs.append(elem[1])
    print("Features:")
    print(features_list)
    print("\nClassifier:")
    print(clf)
    print("\nClassification report:")
    print(metrics.classification_report(labels_test, pred))
    print("\nConfusion matrix:")
    print(metrics.confusion_matrix(labels_test, pred))
    print("#########################\n\n")

# Final classifier and data dump
clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=15),
                                  learning_rate = 0.5, n_estimators= 100)
dump_classifier_and_data(clf, my_dataset, features_list)
#!/usr/bin/python

import pickle
import sys
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn.neighbors import KNeighborsClassifier
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Importing modules for feature scaling and selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from get_features import get_features
from how_many_NaN import how_many_NaN

# Define random seed so that results are the same every time we run the code
np.random.seed(100)

sys.path.append("../tools/")

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".


# Load the dictionary containing the data set
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Replacing NaN values with zeros and dropping 'email_address' column
# First data_dict is converted to pandas dataframe to manipulate NaN value, then it is converted
# back to dictionary

# Converting the pickled Enron data to a pandas dataframe
enron_df = pd.DataFrame(list(data_dict.values()), index=data_dict.keys())

# set the index of df to be the employees series:
employees = pd.Series(list(data_dict.keys()))
enron_df.set_index(employees, inplace=True)

# # Calculate and print how many NaN values are there
# df = how_many_NaN(enron_df)
# print(df)

# Convert data into numeric values, option coerce is used to convert non numeric data to NaN
enron_df = enron_df.apply(lambda x: pd.to_numeric(x, errors='coerce')).copy().fillna(0)

# Dropping column 'email_address'
enron_df.drop('email_address', axis=1, inplace=True)

# Feature engineering
feature_1 = pd.DataFrame(np.zeros((len(enron_df), 1)))
feature_1.set_index(employees, inplace=True)

feature_2 = pd.DataFrame(np.zeros((len(enron_df), 1)))
feature_2.set_index(employees, inplace=True)

feature_3 = pd.DataFrame(np.zeros((len(enron_df), 1)))
feature_3.set_index(employees, inplace=True)
for i in range(len(enron_df)):
    feature_1.iloc[i] = enron_df['bonus'].iloc[i] / enron_df['salary'].iloc[i] if \
        enron_df['salary'][i] != 0.0 else 0.0
    feature_2.iloc[i] = enron_df['from_poi_to_this_person'].iloc[i] / enron_df['to_messages'].iloc[i] if \
        enron_df['to_messages'][i] != 0.0 else 0.0
    feature_3.iloc[i] = enron_df['from_this_person_to_poi'][i] / enron_df['from_messages'].iloc[i] if \
        enron_df['from_messages'][i] != 0.0 else 0.0

enron_df['bonus-to-salary_ratio'] = feature_1
enron_df['from_poi_ratio'] = feature_2
enron_df['to_poi_ratio'] = feature_3

# Define features list

# features_list = get_features(1)  # Include all original features
features_list = get_features(2)  # Include all original features plus 3 engineered features
# features_list = get_features(3) # Only use top features selected by Decision Tree algorithm

# Task 2: Remove outliers
# As explained in attached Jupyter Notebook, the following outliers will be removed from the data set
enron_df.drop('TOTAL', axis=0, inplace=True)
enron_df.drop('THE TRAVEL AGENCY IN THE PARK', axis=0, inplace=True)

# Convert data into numeric values, option coerce is used to convert non numeric data to NaN
enron_df = enron_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Convert dataframe back to dictionary
my_dataset = enron_df.T.to_dict()

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Train test split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

###### Feature pre-processing parameters #####
## Stratified ShuffleSplit cross-validator
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
## Feature scaling
scaler = MinMaxScaler()
## Feature Selection
skb = SelectKBest(f_classif)
## PCA
pca = PCA()


# Task 4: Try a variety of classifiers
######## 1) Decision Tree Classifier #############
# clf_name = "Decision Tree"
# clf = DecisionTreeClassifier()
# pipeline = Pipeline(steps=[("SKB", skb), ("dtree", clf)])
# param_grid = {"SKB__k": range(3, 20),
#               "dtree__criterion": ["gini", "entropy"],
#               "dtree__min_samples_split": [2, 4, 8, 10]}
#
# grid = GridSearchCV(pipeline, param_grid, verbose=0, cv=sss)
# grid.fit(features, labels)
# ## best algorithm
# clf = grid.best_estimator_
# #
# ## Optimum number of features
# k_opt = grid.best_estimator_.steps[0][1].k  # 15 is optimum number of features
# ## Optimum Decision Tree criterion
# dtree_criterion_opt = grid.best_estimator_.steps[1][1].criterion  # "entropy" is optimum criterion
# # Optimum Decision Tree min samples split
# min_samples_split_opt = grid.best_estimator_.steps[1][1].min_samples_split  # 10 is optimum min samples split
#
# ## Obtaining the boolean list showing selected features
# selected_features_bool = grid.best_estimator_.named_steps['SKB'].get_support()
# ## Finding the features selected by SelectKBest
# selected_features_list = [x for x, y in zip(features_list[1:], selected_features_bool) if y]
#
# print "Total number of features selected by SelectKBest algorithm : ", len(selected_features_list)
#
# ## Finding the score of features
# feature_scores = clf.named_steps['SKB'].scores_
# ## Score of features selected by selectKBest
# selected_features_scores = feature_scores[selected_features_bool]
#
# ## Pandas dataframe to store features and their scores and sorting thein in descending order
# features_importance_df = pd.DataFrame({'Features_Selected': selected_features_list, 'Features_score': selected_features_scores})
# features_importance_df.sort_values('Features_score', ascending=False, inplace=True)
# Rank = pd.Series(list(range(1, len(selected_features_list)+1)))
# features_importance_df.set_index(Rank, inplace=True)
# print "Selected features with their corresponding scores", features_importance_df


# clf.fit(features_train, labels_train)
# prediction = clf.predict(features_test)
# importance = clf.feature_importances_
#
# ### Print feature importance into python Dataframe
# dict_importance = {}
# for i in range(len(features_list) - 1):
#     dict_importance[features_list[i + 1]] = importance[i]
# importance_df = pd.DataFrame(dict_importance.items(), columns=['feature', 'importance'])
# importance_df = importance_df.sort_values('importance', ascending=False).set_index('feature')
# print(importance_df)


######## 2) Gaussian Naive Bayes #############
# clf_name = "NB"
clf = GaussianNB()
# clf.fit(features_train, labels_train)
# # prediction = clf.predict(features_test)
# # pipeline = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("NaiveBayes", clf)])
pipeline = Pipeline(steps=[("SKB", skb), ("NaiveBayes", clf)])
# # param_grid = {"SKB__k": range(3, 20),
# #               "PCA__n_components": [2, 3],
# #               "PCA__whiten": [True]}
param_grid = {"SKB__k": range(3, 20)}
#
grid = GridSearchCV(pipeline, param_grid, verbose=0, cv=sss)
grid.fit(features, labels)
# ## best algorithm
clf = grid.best_estimator_

## Obtaining the boolean list showing selected features
selected_features_bool = grid.best_estimator_.named_steps['SKB'].get_support()
## Finding the features selected by SelectKBest
selected_features_list = [x for x, y in zip(features_list[1:], selected_features_bool) if y]

## Finding the score of features
feature_scores = clf.named_steps['SKB'].scores_
## Score of features selected by selectKBest
selected_features_scores = feature_scores[selected_features_bool]

## Pandas dataframe to store features and their scores and sorting thein in descending order
features_importance_df = pd.DataFrame({'Features_Selected': selected_features_list, 'Features_score': selected_features_scores})
features_importance_df.sort_values('Features_score', ascending=False, inplace=True)
Rank = pd.Series(list(range(1, len(selected_features_list)+1)))
features_importance_df.set_index(Rank, inplace=True)
print "Selected features with their corresponding scores", features_importance_df

######## 3) Support Vector Classifier #############
# clf_name = "SVC"
# clf = SVC()
# # clf.fit(features_train, labels_train)
# prediction = clf.predict(features_test)
# pipeline = Pipeline(steps=[("scaling", scaler), ("SKB", skb), ("SVC", clf)])
# param_grid = {"SKB__k": range(3, 20),
#               "SVC__C": [1, 2, 3],
#               "SVC__kernel": ['linear', 'poly', 'rbf']
#               }

# pipeline = Pipeline(steps=[("scaling", scaler), ("SKB", skb), ("PCA", pca), ("SVC", clf)])
# param_grid = {"SKB__k": range(3, 20),
#               "SVC__C": [1, 2, 3],
#               "SVC__kernel": ['linear', 'poly', 'rbf'],
#               "PCA__n_components": [2, 3],
#               "PCA__whiten": [True, False]
#               }
# grid = GridSearchCV(pipeline, param_grid, verbose=0, cv=sss)
# grid.fit(features, labels)
# # ## best algorithm
# clf = grid.best_estimator_
# #
#


# Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)

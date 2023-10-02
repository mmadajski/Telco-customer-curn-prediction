import pandas as pd
import numpy as np
from random import sample, seed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from Utils import calculate_metrics


# Reading data and dropping rows with missing values
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.replace(" ", np.nan, inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data["TotalCharges"] = data["TotalCharges"].astype("float64")
data.drop("customerID", axis="columns", inplace=True)

"""
Dealing with unbalance in target variable.
Each observation in class one is selected twice.
The observations from class zero were selected randomly
so that their number was the same as the number of observations in class one.
"""
zero_class = data[data["Churn"] == "No"].reset_index()
one_class = data[data["Churn"] == "Yes"]

seed(1)
selected_rows = sample([i for i in range(len(zero_class))], 2 * len(one_class))
selected_zero_class = zero_class.iloc[selected_rows]

data = pd.concat([selected_zero_class, one_class, one_class])
data.drop("index", axis="columns", inplace=True)
data.reset_index(inplace=True)

# Converting string variables to binary
variables_to_binary = ("Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn")

for column in variables_to_binary:
    data[column] = pd.Series(np.where(data[column] == "Yes", 1, 0))

data["gender"] = pd.Series(np.where(data["gender"] == "Male", 1, 0))

# One-hot encoding
variables_one_hot_encoding = ("MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod")

for column in variables_one_hot_encoding:
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column).astype("int32")], axis=1)
    data.drop(column, axis="columns", inplace=True)

# Train-test split
answers = data.pop("Churn")
data.drop("index", axis="columns", inplace=True)
data_train, data_test, answers_train, answers_test = train_test_split(data, answers, test_size=0.3, random_state=123)

# Data scaling
scaler = StandardScaler()
scaler.fit(data_train[["tenure", "MonthlyCharges", "TotalCharges"]])

data_train[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(data_train[["tenure", "MonthlyCharges", "TotalCharges"]])
data_test[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(data_test[["tenure", "MonthlyCharges", "TotalCharges"]])

# Logistic Regression
logistic_regression = LogisticRegression(penalty="elasticnet", C=0.01, l1_ratio=0.8, solver="saga")
logistic_regression.fit(data_train, answers_train)

train_prob_lr = logistic_regression.predict_proba(data_train)[:, 1]
test_prob_lr = logistic_regression.predict_proba(data_test)[:, 1]
metrics_train_lr = pd.DataFrame(calculate_metrics(answers_train, train_prob_lr, "lr_train"))
metrics_test_lr = pd.DataFrame(calculate_metrics(answers_test, test_prob_lr, "lr_test"))

fpr, tpr, _ = roc_curve(answers_test, test_prob_lr)
roc_lr = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_lr.to_csv("lr_roc_data.csv", sep=";", decimal=",")

# Variables used in the logistic regression model
regression_coefficients = logistic_regression.coef_
selected_vars = data.columns[[x != 0 for x in regression_coefficients[0]]]
regression_features = pd.DataFrame(selected_vars)
regression_features.to_csv("logistic_regression_features.csv")

# Decision tree.
decision_tree = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=30, random_state=123)
ada_boost_tree = AdaBoostClassifier(n_estimators=30, estimator=decision_tree, learning_rate=1)
ada_boost_tree.fit(data_train, answers_train)

train_prob_tree = ada_boost_tree.predict_proba(data_train)[:, 1]
test_prob_tree = ada_boost_tree.predict_proba(data_test)[:, 1]
metrics_train_tree = pd.DataFrame(calculate_metrics(answers_train, train_prob_tree, "tree_train"))
metrics_test_tree = pd.DataFrame(calculate_metrics(answers_test, test_prob_tree, "tree_test"))

tree_feature_importance = data.columns[[i != 0 for i in ada_boost_tree.feature_importances_]]
tree_feature = pd.DataFrame(tree_feature_importance)
tree_feature.to_csv("tree_features.csv")

metrics_data = pd.concat([metrics_train_lr, metrics_test_lr, metrics_train_tree, metrics_test_tree])
metrics_data.to_csv("models_metrics.csv")

fpr, tpr, _ = roc_curve(answers_test, test_prob_tree)
roc_tree = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_tree.to_csv("tree_roc_data.csv", sep=";", decimal=",")

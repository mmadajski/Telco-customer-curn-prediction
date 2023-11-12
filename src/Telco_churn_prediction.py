import os
import pandas as pd
import numpy as np
from random import sample, seed
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from Utils import get_model_metrics, save_roc_curve_data


# Reading data and dropping rows with missing values
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.replace(" ", np.nan, inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data["TotalCharges"] = data["TotalCharges"].astype("float64")
data.drop("customerID", axis="columns", inplace=True)

"""
Dealing with imbalance in the target variable.
Each observation in class one is selected twice.
The observations in class zero were selected randomly
so that their number is the same as the number of observations in class one.
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
Y = data.pop("Churn")
data.drop("index", axis="columns", inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.3, random_state=123)

# Data scaling
scaler = StandardScaler()
scaler.fit(X_train[["tenure", "MonthlyCharges", "TotalCharges"]])

X_train[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(X_train[["tenure", "MonthlyCharges", "TotalCharges"]])
X_test[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(X_test[["tenure", "MonthlyCharges", "TotalCharges"]])

# Logistic Regression
logistic_regression = LogisticRegression(penalty="elasticnet", C=0.01, l1_ratio=0.8, solver="saga")
logistic_regression.fit(X_train, Y_train)

logistic_regression_metrics = get_model_metrics(logistic_regression, "logistic_regression", X_train, Y_train, X_test, Y_test)

# Variables used in the logistic regression model
regression_coefficients = logistic_regression.coef_
selected_vars = data.columns[[x != 0 for x in regression_coefficients[0]]]
regression_features = pd.DataFrame(selected_vars)

# Decision tree
decision_tree = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=30, random_state=123)
ada_boost_tree = AdaBoostClassifier(n_estimators=30, estimator=decision_tree, learning_rate=1)
ada_boost_tree.fit(X_train, Y_train)

tree_metrics = get_model_metrics(ada_boost_tree, "ada_boost_tree", X_train, Y_train, X_test, Y_test)

# Decision tree feature importance
tree_feature_importance = data.columns[[i != 0 for i in ada_boost_tree.feature_importances_]]
tree_feature = pd.DataFrame(tree_feature_importance)

# Saving metrics and used features
if not os.path.exists("..\\metrics"):
    os.makedirs("..\\metrics")

metrics_data = pd.concat([tree_metrics, logistic_regression_metrics])
metrics_data.to_csv("..\\metrics\\models_metrics.csv", sep=";", decimal=",")

save_roc_curve_data(logistic_regression, "logistic_regression_roc_data", X_test, Y_test)
save_roc_curve_data(ada_boost_tree, "ada_boost_tree_roc_data", X_test, Y_test)

tree_feature.to_csv("..\\metrics\\tree_features.csv")
regression_features.to_csv("..\\metrics\\logistic_regression_features.csv")
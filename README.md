# Telco customer churn prediction

Acquiring customers is a difficult and expensive task,
which is why companies don't want their customers to leave after they start
using their services. Therefore, identifying customers who will leave is essential,
because it may allow a company to take measures to retain them. In this project,
I created models that are able to both classify customers who leave and select the most important variables in the dataset. 

Technologies used: Python, Sklearn, Pandas, PowerBI

Project based on Telco Customer Churn data: [telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Getting familiar with the data

Before creating the models, I sought to better understand the data and gain insight into customer decisions.
I have put the most interesting of the extracted information on a dashboard created using PowerBI.

![](https://github.com/mmadajski/Telco-customer-curn-prediction/blob/main/Dashboards/Telco_Churn_analysis.png)

On the dashboard above we can see that customers using fiber optics or pay with electronic check have much higher chance of leaving the company. 
This may be due to the poor quality of services served. We should also encourage customers to sign contracts other than monthly, because a very high percentage of those who sign such a contract leave. 

---

## Data preparation 

The dataset contained 20 variables and 7043 observations, 11 of which contained missing data.
In the dataset were 4 binary variables, 
10 categorical variables and 3 variables were continuous. 
The target variable named "Churn" is a binary variable describing whether a customer has left the company (here 1 means that customer left).

The categorical variables were recoded to binary using one-hot encoding. Then data was split into train and test datasets,
and standardized.

---

## Models 

Due to high number of variables (41 variables after one-hot encoding) it was important to select the most important variables.

Therefor two model were selected: penalized logistic regression and decision tree.

---

### Penalized Logistic Regression

The logistic regression model was fitted using a elastic network, a C-parameter of 0.01 (the C-parameter is the inverse of the regularization strength) and a l1 ratio of 0.8 (the l1 ration is the mixing parameter of the elastic network).

---

### Decision Tree

The decision tree was fitted using gini impurity, a maximum depth of 5 and a minimum number of observations in a leaf of 30. 
In addition, to improve tree performance, the AdaBoost algorithm with 30 estimators and a learning rate of 1 was used. 

---

# Performance comparison 

#### Train data
|   | Logistic Regression | Decision Tree |
|---|---|---|
|Accuracy| 0,76|0,89|
|Recall| 0,8 | 0,91
|Sensitivity| 0,73 |0,87
|Precision| 0,75 |0,88
|Specificity| 0,73 |0,87
|AUC| 0,84 |0,97

---

#### Test data
|   | Logistic Regression | Decision Tree |
|---|---|---|
|Accuracy| 0,76|0,78|
|Recall| 0,78| 0,81
|Sensitivity| 0,74 |0,74
|Precision| 0,74 |0,76
|Specificity| 0,74 |0,74
|AUC| 0,84 |0,85

The decision tree used 32 variables in the prediction, while the logistic regression model used only 10.
---

## Conclusion 

Even though metrics such as accuracy and recall are slightly higher for the decision tree, logistic regression is a better model because it was able to make predictions using only 10 variables compared to the 32 variables used by the decision tree. In addition the decision tree is severly overfitted.




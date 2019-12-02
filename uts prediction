# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('uts.csv')
print(dataset)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 3].values
print(X)
print("\n")
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("The depending training set is: \n")
print(X_train)
print("The depending test set is: \n")
print(X_test)
print("The independent training set is: \n")
print(y_train)
print("The independent test set is: \n")
print(y_test)

from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Fitting Multiple Linear Regression To The Training Set
from sklearn.linear_model import LinearRegression
regressor_mlr = LinearRegression()
regressor_mlr.fit(X_train, y_train)

#Predicting the Test set Results
y_pred_mlr = regressor_mlr.predict(X_test)

#Printing the results
print(y_pred_mlr)

#Predicting the accuracy of MLR
mlr_model_pred=regressor_mlr.predict(X_test)
acc_mlr_model=round(regressor_mlr.score(X_train, y_train)*100,2)
print(acc_mlr_model)

#Fitting Stochastic gradient descent(SGD)
from sklearn.linear_model import SGDClassifier
regressor_sgd=SGDClassifier()
regressor_sgd.fit(X_train,y_train)

#Predicting the Test set Results
y_pred_sgd = regressor_sgd.predict(X_test)

#Printing the results
print(y_pred_sgd)

#Predicting the accuracy of SGD
sgd_model_pred=regressor_sgd.predict(X_test)
acc_sgd_model=round(regressor_sgd.score(X_train, y_train)*100,2)
print(acc_sgd_model)

#Fitting logistic regression to training set
from sklearn.linear_model import LogisticRegression
regressor_log=LogisticRegression()
regressor_log.fit(X_train,y_train)

#Predicting the Test set Results
y_pred_log = regressor_log.predict(X_test)

#Printing the results
print(y_pred_log)

#Predicting the accuracy of Logistic regression
log_model_pred=regressor_log.predict(X_test)
acc_log_model=round(regressor_log.score(X_train, y_train)*100,2)
print(acc_log_model)

#Ftting KNN to training set
from sklearn.neighbors import KNeighborsClassifier
regressor_knn=KNeighborsClassifier()
regressor_knn.fit(X_train,y_train)

#Predicting the Test set Results
y_pred_knn = regressor_knn.predict(X_test)

#Printing the results
print(y_pred_knn)

#Predicting the accuracy of KNN
knn_model_pred=regressor_knn.predict(X_test)
acc_knn_model=round(regressor_knn.score(X_train, y_train)*100,2)
print(acc_knn_model)

#Fitting SVM to training set
from sklearn.svm import SVC, LinearSVC
regressor_svm=LinearSVC()
regressor_svm.fit(X_train,y_train)

#Predicting the Test set Results
y_pred_svm = regressor_svm.predict(X_test)

#Printing the results
print(y_pred_svm)

#Predicting the accuracy of KNN
svm_model_pred=regressor_svm.predict(X_test)
acc_svm_model=round(regressor_svm.score(X_train, y_train)*100,2)
print(acc_svm_model)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_svm = sc.fit_transform(X_train)
X_test_svm = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_rbf = SVC(kernel = 'rbf', random_state = 0)
classifier_rbf.fit(X_train_svm, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_svm)
print(y_pred)
print("\n")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Predicting the accuracy of SVM
rbf_model_pred=classifier_rbf.predict(X_test)
acc_rbf_model=round(classifier_rbf.score(X_train, y_train)*100,2)
print(acc_rbf_model)

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_des = DecisionTreeRegressor(random_state = 0)
regressor_des.fit(X_train, y_train)

#Predicting the Test set Results
y_pred_des = regressor_des.predict(X_test)

#Printing the results
print(y_pred_des)

#Predicting the accuracy of Decision tree
des_model_pred=regressor_des.predict(X_test)
acc_des_model=round(regressor_des.score(X_train, y_train)*100,2)
print(acc_des_model)

# Fitting the Random forest Regression Model to the training set
from sklearn.ensemble import RandomForestClassifier
regressor_rfr=RandomForestClassifier()
regressor_rfr.fit(X_train,y_train)

#Predicting the Test set Results
y_pred_rfr = regressor_rfr.predict(X_test)

#Printing the results
print(y_pred_rfr)

#Predicting the accuracy of Random forest
rfr_model_pred=regressor_rfr.predict(X_test)
acc_rfr_model=round(regressor_rfr.score(X_train, y_train)*100,2)
print(acc_rfr_model)

results = pd.DataFrame({
    'Model': ['Multiple linear regression', 'Stochastic gradient descent (SGD)', 'Logistic Regression', 
              'KNN', 'Linear SVM', 'RBF SVM', 'Decision Tree', 'Random forest'],
    'Score': [acc_mlr_model, acc_sgd_model, acc_log_model, 
              acc_knn_model, acc_svm_model, acc_rbf_model, acc_des_model,
              acc_rfr_model]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df

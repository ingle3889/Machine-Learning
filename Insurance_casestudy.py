# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 14:14:48 2022

@author: TiaaUser
"""
""" https://docs.microsoft.com/en-us/sysinternals/downloads/sysinternals-suite"""
""" Logisitc Regression with Insurance dataset"""

""" Learn design path and algorithm for better sfoter

 Design pattern of gangs of 4 :book Name """

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def Log(path):
    df = pd.read_csv(path)
    print("_"* 50)
    print("First few entries of dataset")
    print(df.head())
    
    print("_"* 50)
    
    plt.scatter(df.Age,df.Bought_insurance, marker='+', color ='red')
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(df[['Age']], df.Bought_insurance, train_size=0.5)
    
    print("Independednt variable fro training")
    print(X_train)
    
    print("Independednt variable fro training")
    print(y_train)
    
    print("Independednt variable fro training")
    print(X_test)
    
    print("Independednt variable fro training")
    print(y_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("_"* 50)
    y_predicted = model.predict(X_test)
    print("Predicted dependent variable")
    print(y_predicted)
    
    print("Expected dependent variable")
    print(y_test)
    
    print("_"* 50)
    data = model.predict_proba(X_test)
    print("Probability of above model is :")
    print(data)
    
    print("_"* 50)
    print("Classification report of logistic Regression is :")
    print(classification_report(y_test, y_predicted))
    
    print("_"* 50)
    print("Classification report of logistic Regression is :")
    print(classification_report(y_test, y_predicted))
    
    print("_"* 50)
    print("Confusion_matrix report of logistic Regression is :")
    print(confusion_matrix(y_test, y_predicted))
    
    print("_"* 50)
    print("Accuracy_score report of logistic Regression is :")
    print(accuracy_score(y_test, y_predicted))
    
            
         
def main():
    print("_"* 50)
    
    print("Supervised ML")
    print("_"* 50)
    print("Logistic regression on insurance set")
    print("_"* 50)
    Log("insurance.csv")
    
if __name__ == '__main__':
    main()    
    
    
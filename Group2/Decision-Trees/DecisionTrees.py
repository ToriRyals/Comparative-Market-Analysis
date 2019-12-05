# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline - leaves syntax error
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
#####################
#Libraries for visuals 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import collections

#### readFile = dataset

#reads all columns from original file
readOriginalFile = pd.read_csv("prepared_dataset.csv")
print(readOriginalFile.shape) #prints the number of rows and columns from the csv file
print(readOriginalFile.head()) #prints data from csv file

#reads all columns from original file except city, state & zipcode
readShortFile = pd.read_csv("prepared_dataset.csv", usecols = ['beds', 'baths', 'sq__ft', 'price'])
print(readShortFile.shape) #prints the number of rows and columns from the csv file
print(readShortFile.head()) #prints data from csv file



X = readShortFile.drop('beds', axis=1) #reads only bath, sqft and price column -> features

y = readShortFile['beds'] #reads only bed column -> target variable 


#splits 20% of data into test set and 80% into training  -> random split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

####################################################################
#Graph

feature_names = ['baths', 'sq__ft', 'price']
dotfile = open("Desktop\dtree2.dot", 'w')
feature_names = ['baths', 'sq__ft', 'price']
tree.export_graphviz(classifier, out_file=dotfile,  
                     feature_names=feature_names ,  
                     class_names=True)
dotfile.close()

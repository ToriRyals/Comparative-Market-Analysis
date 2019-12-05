import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from IPython.display import Image  
import pydotplus
import collections

dataset = pd.read_csv('dataset1.csv')
dataset.head()
readShortFile = pd.read_csv("dataset1.csv", usecols = ['beds', 'baths', 'sq__ft', 'price'])

X = readShortFile.drop('beds', axis=1) #reads only bath, sqft and price column -> features

y = readShortFile['beds'] #reads only bed column -> target variable  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test, y_pred))

#Graph

feature_names = ['baths', 'sq__ft', 'price']
dotfile = open("Desktop\dtree2.dot", 'w')
feature_names = ['baths', 'sq__ft', 'price']
tree.export_graphviz(classifier[0], out_file=dotfile,  
                     feature_names=feature_names ,  
                     class_names=True)
dotfile.close()

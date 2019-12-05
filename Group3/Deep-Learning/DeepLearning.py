import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

# Get the property data 
readPropertyData = pd.read_csv('shortData.csv') 

# Get the property ratings
readRatings = pd.read_csv('customerRatings.csv') 

#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(readRatings, test_size=0.2, random_state=42)

#Variables propertyID and ratings used 
n_propertyid = len(readRatings.Property_ID.unique())
n_customers = len(readRatings.Customer.unique())

# creating propertyID embedding path
propertyID_input = Input(shape=[1], name="propertyID-Input")
propertyID_embedding = Embedding(n_propertyid+1, 5, name="propertyID-Embedding")(propertyID_input)
propertyID_vec = Flatten(name="Flatten-PropertyID")(propertyID_embedding)

# creating user/customer embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_customers+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([propertyID_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model = Model([user_input, propertyID_input], out)
model.compile('adam', 'mean_squared_error')

from keras.models import load_model

if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.user, train.propertyID], train.Rating, epochs=5, verbose=1)
    model2.save('regression_model2.h5')
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")

# Extract embeddings
propertyID_em = model.get_layer('Property-Embedding')
propertyID_em_weights = propertyID_em.get_weights()[0]
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
pca_result = pca.fit_transform(propertyID_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])


#Prediction of top 5 properties for a user

# Creating dataset for making recommendations for the first user
property_data = np.array(list(set(readRatings.Property_ID)))
property_data[:5]
user = np.array([1 for i in range(len(property_data))])
user[:5]
predictions = model.predict([user, property_data])

predictions = np.array([a[0] for a in predictions])

recommended_property_ids = (-predictions).argsort()[:5]
# print predicted scores
recommended_property_ids

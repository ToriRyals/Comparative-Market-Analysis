import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Get the property data 
readPropertyData = pd.read_csv("shortData.csv") 
  
# Get the property ratings
readRatings = pd.read_csv('customerRating.csv') 


# create customer vs property matrix for ratings

property_customer_matrix = readRatings.pivot(index='Property_ID', columns='Customer', values='Rating').fillna(0)
#map property to street name
property_to_street = {
    property: i for i, property in 
    enumerate(list(readPropertyData.set_index('Property_ID').loc[property_customer_matrix.index].street))
}
# transform matrix to scipy sparse matrix
property_customer_matrix_sparse = csr_matrix(property_customer_matrix.values)



print("Customer vs Property")
print("Numbers indicate the ratings each customer gives each property\n")
print(property_customer_matrix)



#counting the number of ratings

readPropertyData_cnt = pd.DataFrame(readRatings.groupby('Property_ID').size(), columns=['count'])
print("\nHow many ratings each property has:\n")
print(readPropertyData_cnt.head())

# get number of ratings given by every user 
customers_cnt = pd.DataFrame(readRatings.groupby('Customer').size(), columns=['count'])
print("\nHow many ratings each customer made:\n")
print(customers_cnt.head())


#make an object for the NearestNeighbors Class.
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit the dataset
model_knn.fit(property_customer_matrix_sparse)

#checks user input and returns the closest matcch
#Ex. customer input "High St" vs "High Street"
def fuzzy_matching(mapper, fav_property, verbose=True):
    match_tuple = []
    # get match
    for street, idx in mapper.items():
        ratio = fuzz.ratio(street.lower(), fav_property.lower())
        if ratio >= 60:
            match_tuple.append((street, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('\nOops! No match is found')
        return
    if verbose:
        print('\nFound possible matches found for: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

#computes recommendations for similar properties
def make_recommendation(model_knn, data, mapper, fav_property, n_recommendations):
    
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input property:', fav_property)
    idx = fuzzy_matching(mapper, fav_property, verbose=True)
    
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:\n'.format(fav_property))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

my_favorite = '3526 HIGH ST'

make_recommendation(
    model_knn=model_knn,
    data=property_customer_matrix_sparse,
    fav_property=my_favorite,
    mapper=property_to_street,
    n_recommendations=3)




property_to_street





# import pandas library 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Get the property data 
readPropertyData = pd.read_csv("shortData.csv") 
  
# Get the property ratings
readRatings = pd.read_csv('customerRating.csv') 

# Merge data and export to csv 
newData = pd.merge(readPropertyData, readRatings, on='Property_ID') 
newData.to_csv("newData.csv")


# creating dataframe with average rating values for each property and export to csv
ratings = pd.DataFrame(newData.groupby('street')['Rating'].mean().sort_values(ascending=False))  
print(ratings)
ratings.to_csv("GroupedRatings.csv")

#separte columns (street name and ratings)
readGroupedRatings = pd.read_csv("GroupedRatings.csv")
streetNames=(readGroupedRatings['street'])
ratingValues=(readGroupedRatings['Rating'])

#create horizontal bar graph
objects = streetNames
y_pos = np.arange(len(streetNames))
performance = ratingValues
plt.figure(figsize=(12,4))
plt.barh(y_pos, performance, align='center', alpha=0.5,)
plt.yticks(y_pos, objects, size=5)
plt.ylabel('Property Street')
plt.xlabel('Average Rating')
plt.title('Average Property Ratings')
out_png2 = "Desktop\CollabFilt.png"
plt.savefig(out_png2, dpi=175)
plt.show()

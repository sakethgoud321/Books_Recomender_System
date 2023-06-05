#!/usr/bin/env python
# coding: utf-8

# ### Business Objective:
# Generate the features from the dataset and use them to recommend the books accordingly to the users.
# 
# ### Data Set
# The Book-Crossing dataset comprises 3 files.
# 
# #### 1. Users -
# Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.
# 
# #### 2. Books -
# Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.
# 
# #### 3.Ratings -
# Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

# In[63]:


# Importing Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[64]:


# Loading Datasets
books_df = pd.read_csv("Books.csv",on_bad_lines='skip')
users_df = pd.read_csv("Users.csv",on_bad_lines='skip')
ratings_df = pd.read_csv("Ratings.csv",on_bad_lines='skip')


# In[65]:


books_df.head()


# In[66]:


# Rename the columns
books = books_df.rename(columns={
    'ISBN': 'isbn',
    'Book-Title': 'title',
    'Book-Author': 'author',
    'Year-Of-Publication': 'year',
    'Publisher': 'publisher',
    'Image-URL-S': 'image_url_s',
    'Image-URL-M': 'image_url_m',
    'Image-URL-L': 'image_url_l'
})

# Check the new column names
print(books.columns)


# In[67]:


users_df.head()


# In[68]:


# Rename the columns
users = users_df.rename(columns={
    'User-ID': 'userid',
    'Location': 'location',
    'Age': 'age',
})

# Check the new column names
print(users.columns)


# In[69]:


ratings_df.head()


# In[70]:


# Rename the columns
ratings = ratings_df.rename(columns={
    'User-ID': 'userid',
    'ISBN': 'isbn',
    'Book-Rating': 'rating',
})

# Check the new column names
print(ratings.columns)


# ## Descriptive Statistics

# In[71]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[72]:


books.info()


# In[73]:


books.isnull().sum()


# In[74]:


books.duplicated().sum()


# In[75]:


users.info()


# In[76]:


users.isnull().sum()


# In[77]:


# Number of missing values in age column
num_missing=users['age'].isnull().sum()
num_missing


# In[78]:


# missing values in percentage
percent_missing=(num_missing/len(users))*100
print(percent_missing)


# In[79]:


users.duplicated().sum()


# In[80]:


ratings.info()


# In[81]:


ratings.isnull().sum()


# In[82]:


ratings.duplicated().sum()


# ## Exploratory Data Analysis

# ### Books

# In[83]:


# Change the data type of the year column to int
# Change the data type of the publication_year column to int
books['year'] = pd.to_numeric(books['year'], errors='coerce')
books = books[~books['year'].isna()]
books['year'] = books['year'].astype(int)


# In[84]:


books.year.unique()


# In[85]:


books.year.describe()


# In[86]:


books.isna().sum()


# - There is 1 null value in Author column
# - 2 missing values in publisher column

# In[87]:


author_na=pd.isnull(books['author'])
books[author_na]


# In[88]:


# finding the missing auther name from external refference and replacing
books.loc[187689, 'author'] = 'Larissa Anne Downes'


# In[89]:


publisher_na=pd.isnull(books["publisher"])
books[publisher_na]


# In[90]:


# finding the missing publisher name from external refference and replacing
books.loc[128890,'publisher']='Novelbooks Inc'
books.loc[129037,'publisher']='Bantam'


# In[91]:


books.isna().sum()


# In[92]:


# Check the distribution of the year of publication
plt.hist(books['year'], bins=range(1920, 2030, 10),color="violet")
plt.xlabel('Year of Publication')
plt.ylabel('Frequency')
plt.show()


# In[93]:


# Top 10 authors
print("Top 10 Authors-")
print("\n")

top_authors = books['author'].value_counts().head(10)
print(top_authors)

print("\n")

# Top 10 publishers
print("Top 10 Publishers-")
print("\n")

top_publishers = books['publisher'].value_counts().head(10)
print(top_publishers)


# In[94]:


# Identify popular authors
import matplotlib.pyplot as plt

# Top 15 authors
popular_authors = books['author'].value_counts().head(15)
popular_authors.plot.barh()
plt.xlabel('Frequency')
plt.title('Top 15 Authors')

# Reverse the y-axis to show the bars in descending order
plt.gca().invert_yaxis()
    
plt.show()


# In[95]:


# Top 10 Publishers
popular_publishers = books['publisher'].value_counts().head(10)
popular_publishers.plot.barh()
plt.xlabel('Frequency')
plt.title('Top 10 Publishers')

# Reverse the y-axis to show the bars in descending order
plt.gca().invert_yaxis()
    
plt.show()


# ### Users

# In[96]:


# Check the distribution of age
users['age'].hist(bins=range(0, 110, 10),color="green")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[97]:


median_age=users['age'].median()
median_age


# In[98]:


# Replacing all null values with median
users['age']=users['age'].fillna(median_age)
users['age'].isnull().sum()


# In[99]:


def age_group(age):
    if age<18:
        x='Children'
    elif age>=18 and age<35:
        x='Youth'
    elif age>=35 and age<65:
        x='Adults'
    else:
        x='Senior Citizens'
    return x


# In[100]:


users['age_group']=users['age'].apply(lambda x: age_group(x))
users.head()


# In[101]:


plt.figure(figsize=(15,7))
sns.countplot(y='age_group',data=users)
plt.title('Age Distribution')


# In[102]:


users.location.unique()


# In[103]:


# Users by Location
users_by_location = users['location'].value_counts().head(15)
users_by_location.plot(kind='bar',color="red")
plt.xlabel('Location')
plt.ylabel('Number of Users')

plt.show()


# In[104]:


import re

# Define a function to extract country name
def extract_country(location):
    country = re.findall(r'\,+\s?(\w*\s?\w*)\"*$', location)
    if country:
        return country[0]
    else:
        return None

# Apply the function to the location column
users['country'] = users['location'].apply(extract_country)


# In[105]:


users.isnull().sum()


# In[106]:


users['country']=users['country'].astype('str')


# In[107]:


# Users by Country
users_by_location = users['country'].value_counts().head(15)
users_by_location.plot(kind='bar',color="red")
plt.xlabel('Country')
plt.ylabel('Number of Users')

plt.show()


# ### Ratings

# In[108]:


ratings.rating.value_counts()


# In[109]:


# Distribution of ratings
sns.countplot(x='rating', data=ratings)
plt.show()


# ## Relation Between Book and Ratings

# In[110]:


# Merge the datasets
data = pd.merge(ratings, books, on='isbn')
data


# In[111]:


data.isnull().sum()


# In[112]:


# explore relationships between columns

# Correlation matrix
corr = data[['rating', 'year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[113]:


ratings_df = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings_df.rename({'rating':'Mean Ratings'}, axis=1 , inplace =True)
ratings_df['No of times Rated'] = pd.DataFrame(data.groupby('title')['rating'].count())
ratings_df


# In[114]:


plt.figure(figsize=(10,4), dpi=100)
sns.jointplot(x='Mean Ratings',y='No of times Rated',data=ratings_df,alpha=0.5)
plt.show()


# In[115]:


# Create a scatter plot
import matplotlib.pyplot as plt

# Filter the dataframe to include only books published between 1900 and 2025
df_filtered = data[(data["year"] >= 1900) & (data["year"] <= 2025)]

# Create a scatter plot of ratings versus year
plt.scatter(df_filtered["year"], df_filtered["rating"])
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()


# In[116]:


# Group the data by decade
data['decade'] = pd.cut(data['year'], bins=range(1900, 2030, 10))

# Create a box plot for each decade
data.boxplot(column='rating', by='decade', figsize=(15, 6))

# Set the axis labels and title
plt.xlabel("Decade")
plt.ylabel("Rating")
plt.title("Distribution of Ratings by Decade")

# Show the plot
plt.show()


# In[117]:


popular_books = pd.DataFrame(data.groupby('title')['rating'].count())
most_popular = popular_books.sort_values('rating', ascending=False)
most_popular.rename({'rating':'No of times Rated'}, axis=1, inplace=True)
most_popular.head(10)


# In[118]:


plt.figure(figsize=(12,8), dpi=200)
most_popular.head(30).plot(kind = "bar",figsize=(12,8))
plt.title('Most Popular Books',  fontsize = 16, fontweight = 'bold')
plt.show()


# In[119]:


# Heatmap
data['decade'] = pd.cut(data['year'], bins=range(1900, 2030, 10))
rating_freq = data.pivot_table(index='rating', columns='decade', values='isbn', aggfunc='count')

cmap = sns.cm.rocket_r 
sns.heatmap(rating_freq, cmap=cmap)
plt.title("Frequency of Ratings Over Years")
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()


# In[120]:


# Final Data
final_data = pd.merge(users, data, on='userid')
final_data


# In[121]:


final_data.shape


# In[122]:


final_data.isnull().sum()


# In[123]:


final_df = final_data.drop(['decade', 'age_group', 'image_url_s', 'image_url_l'], axis=1)
final_df


# In[124]:


final_df.to_csv('final_data.csv', index=False)


# In[ ]:





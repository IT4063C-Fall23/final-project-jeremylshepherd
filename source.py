#!/usr/bin/env python
# coding: utf-8

# # Affect of Marijuana legalization on crime rates
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 
# Evaluate the impact of marijuana legalization on other areas of crime through comparative analysis to establish/refute correlation.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# Does marijuana legalization lead to an increase in other areas of criminality?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# I suspect there will be some increases in other areas such as impaired driving. I do not have a strong inclination towards the degree it is impacted.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# [Seattle Crime stats](https://www.kaggle.com/datasets/city-of-seattle/seattle-crime-stats)
# [Denver Crime stats](https://www.kaggle.com/datasets/paultimothymooney/denver-crime-data/data)
# [Kansas City Crime](https://www.kaggle.com/datasets/riteshkadakoti/crime-dataset-kansas)
# 
# *How are you going to relate these datasets?*
# My working plan to compare 2 large cities where marijuana was first legalized (2012) for recreational use. I plan compare pre and post legalization crime trends. Then further compare/contrast that with a large city that had not (until this year) legalized recreational use. I am still ruminating on the mechanics of how to best do that.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[1]:


# Start your code here
# Not sure what the expectation is for this block
import numpy as np
import pandas as pd
import datetime
from scipy.stats import trim_mean

# Configure pandas to display 500 rows; otherwise it will truncate the output
pd.set_option('display.max_rows', 500)

import os

# Scikit Learn imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

denver_crime_df = pd.read_csv('data/denver_crime.csv')
kc_crime_df = pd.read_csv('data/KCcrime2010To2018.csv', low_memory=False)
seattle01_crime_df = pd.read_csv('data/seattle-crime-stats-by-1990-census-tract-1996-2007.csv')
seattle02_crime_df = pd.read_csv('data/seattle-crime-stats-by-police-precinct-2008-present.csv')

denver_crime_df.head()
kc_crime_df.head()
seattle01_crime_df.head()
seattle02_crime_df.head()


# ### Exploratory Data Analysis
# 
# Data visualizations to evaluate the data in order to form conclusions about whether marijuana legalization had a tangential impact on other aspects of crime.
# I am paying particular attention to the 2-3 year period before and after legalization.

# In[2]:


# Get the shapes of the data
display(denver_crime_df.shape)
display(seattle01_crime_df.shape)
display(seattle02_crime_df.shape)
display(kc_crime_df.shape)


# In[3]:


# Look at the columns and compare the presentation of the data
display(denver_crime_df.info())
display(seattle01_crime_df.info())
display(seattle02_crime_df.info())
display(kc_crime_df.info())


# ### Intitial thoughts
# The data from Seattle 01 and Kansas City do not appear to provide much in the way of usable data. Denver and Seattle 02 show some promise. The datasets are structured differently but I may be able to extract the necessary data from them to answer the question I am researching. I will need to try to find another comparison city for evaluation against the cities with legalized marijuana.

# In[4]:


# Check for duplicated records for Denver
denver_crime_df.duplicated().sum()


# In[5]:


# Check for duplicated records for Seattle
seattle02_crime_df.duplicated().sum()


# In[6]:


# Drop duplicates from Seattle and rename
display(seattle02_crime_df.shape)
seattle_crime_df = seattle02_crime_df.drop_duplicates()
display(seattle_crime_df.shape)


# In[7]:


# Check for missing values for Denver
display(denver_crime_df.isna().sum())
display(seattle_crime_df.isna().sum())


# ### Data cleaned and ready for use
# The missing values are immaterial to the scope of analysis and can be safely ignored. The four duplicates from Seattle have been dropped.

# ### Statistical Summary of the Data
# Below is the statistical summary of the data for Denver and Seattle.

# In[8]:


denver_crime_df.describe()


# In[9]:


seattle_crime_df.describe()


# ### Data visualizations
# Display a representation of the data to evaluate best way to evaluate it.

# In[10]:


denver_crime_df.hist(figsize=(20,15), bins=50)


# In[11]:


seattle_crime_df.hist(figsize=(20,15), bins=50)


# In[12]:


seattle_crime_df['CRIME_TYPE'].hist(figsize=(20,15), bins=50)


# In[13]:


denver_crime_df['offense_category_id'].hist(figsize=(20,15), bins=50)


# In[14]:


simplified_denver_crime_df = denver_crime_df.drop(['incident_id','offense_id','offense_code','offense_code_extension','first_occurrence_date',
                                                 'last_occurrence_date','incident_address','geo_x','geo_y','geo_lon','geo_lat','district_id',
                                                 'precinct_id','neighborhood_id','is_crime','is_traffic','victim_count'], axis=1)

simplified_denver_crime_df['offense_category_id'].value_counts().plot(kind='barh')


# In[15]:


seattle_scatter_mat = pd.plotting.scatter_matrix(seattle_crime_df, figsize=(20,15))
seattle_scatter_mat


# In[16]:


seattle_crime_df['CRIME_TYPE'].value_counts().plot(kind='barh')


# ### First Impressions/Conclusions
# Scatter Matrixes were not particularly helpful. I see that the primary data I want is the total crime type by year. I will need to eliminate unneeded columns and so some data transformation to make the sets more useful. I have trimmed down known extraneous columns from denver just reduce the load on the data viz libraries. I will refine further during the next phase.

# ## Machine Learning and Regression
# ### Data transformation
# I will add a column to each dataset exclusively for the year of the offense. This should allow me to observe the rate of change, if any, for each type year over year. The datasets have different categories of crimes that do not necessarily correspond to each other. However, I do not see this necessarily as a hindrance as the real intent is to see if there are any changes in the respective categories.

# In[17]:


# Transform Denver data
simplified_denver_crime_df['reported_datetime'] = pd.to_datetime(simplified_denver_crime_df['reported_date'])
simplified_denver_crime_df.head()
simplified_denver_crime_df['year'] = simplified_denver_crime_df['reported_datetime'].dt.year
simplified_denver_crime_df.head()


# In[18]:


seattle_crime_df['REPORT_DATETIME'] = pd.to_datetime(seattle_crime_df['REPORT_DATE'])
seattle_crime_df.head()
seattle_crime_df['YEAR'] = seattle_crime_df['REPORT_DATETIME'].dt.year
seattle_crime_df.head()


# ### Remove all superflous columns
# Remove unnecessary columns from the datasets to maximize focus on important data.

# In[19]:


denver_crime_df_v2 = simplified_denver_crime_df.drop(['offense_type_id', 'reported_date',
       'reported_datetime'], axis=1)
denver_crime_df_v2.head()


# In[20]:


seattle_crime_df_v2 = seattle_crime_df.drop(['Police Beat', 'CRIME_DESCRIPTION', 'STAT_VALUE', 'REPORT_DATE',
                                             'Sector', 'Precinct', 'Row_Value_ID', 'REPORT_DATETIME'] , axis=1)
seattle_crime_df_v2.head()


# In[21]:


seattle_grouping_and_counts = seattle_crime_df_v2.groupby('YEAR')['CRIME_TYPE'].value_counts()
seattle_grouping_and_counts.plot(x='YEAR', kind='bar', stacked=False, title='Seattle Criminal Incidents by Year', figsize=(150, 75))


# In[22]:


denver_grouping_and_counts = denver_crime_df_v2.groupby('year')['offense_category_id'].value_counts()
denver_grouping_and_counts.plot(x='year', kind='bar', stacked=False, title='Denver Criminal Incidents by Year', figsize=(100, 50))


# ## Machine Learning
# 
# Split into train and test sets. Given this is for demonstration purposes only. I will limit it to the Seattle dataset.
# 
# 

# In[58]:


from sklearn.model_selection import train_test_split

# create train/test set
seattle_train_set, seattle_test_set = train_test_split(seattle_crime_df_v2, test_size=0.2, random_state=42)
seattle_train_set.head()


# ## Modeling Data
# Using Machine Learning to build a predictive model.
# 
# Given the simple nature of the data, I am using a LinearRegression model. Modeling is not really relevant to my project, but I include it to demonstrate using the tools.

# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# handle the categorical data
seattle_categories = seattle_train_set[['CRIME_TYPE']]
one_hot_encoder = OneHotEncoder()
seattle_train_categories = one_hot_encoder.fit_transform(seattle_categories)
display(seattle_train_categories.toarray())
display(one_hot_encoder.categories_)

# Trying to convert the Groupby variable into a dataframe but it doesn't really work. I end up with a single column of count instead of [YEAR, CRIME_TYPE, COUNT]. Not sure how to accomplaish that which has me stuck to this point.



# ## Resources and References
# *What resources and references have you used for this project?*
# I used Kaggle to source the datasets

# In[25]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


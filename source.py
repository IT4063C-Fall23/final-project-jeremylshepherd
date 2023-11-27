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
from scipy.stats import trim_mean

# Configure pandas to display 500 rows; otherwise it will truncate the output
pd.set_option('display.max_rows', 500)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
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


# In[17]:


simplified_denver_crime_df = denver_crime_df.drop(['incident_id','offense_id','offense_code','offense_code_extension','first_occurrence_date',
                                                 'last_occurrence_date','incident_address','geo_x','geo_y','geo_lon','geo_lat','district_id',
                                                 'precinct_id','neighborhood_id','is_crime','is_traffic','victim_count'], axis=1)

simplified_denver_crime_df['offense_category_id'].value_counts().plot(kind='barh')


# In[15]:


seattle_scatter_mat = pd.plotting.scatter_matrix(seattle_crime_df, figsize=(20,15))
seattle_scatter_mat


# In[18]:


seattle_crime_df['CRIME_TYPE'].value_counts().plot(kind='barh')


# # First Impressions
# Scatter Matrixes were not particularly helpful. I see that the primary data I want is the total crime type by year. I will need to eliminate unneeded columns and so some data transformation to make the sets more useful. I have trimmed down known extraneous columns from denver just reduce the load on the data viz libraries. I will refine further during the next phase.

# ## Resources and References
# *What resources and references have you used for this project?*
# I used Kaggle to source the datasets

# In[19]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


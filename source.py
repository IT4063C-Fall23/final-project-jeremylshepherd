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

# In[5]:


# Start your code here
# Not sure what the expectation is for this block
import numpy as np
import pandas as pd

denver_crime_df = pd.read_csv('data/denver_crime.csv')
kc_crime_df = pd.read_csv('data/KCcrime2010To2018.csv', low_memory=False)
seattle01_crime_df = pd.read_csv('data/seattle-crime-stats-by-1990-census-tract-1996-2007.csv')
seattle02_crime_df = pd.read_csv('data/seattle-crime-stats-by-police-precinct-2008-present.csv')

denver_crime_df.head()
kc_crime_df.head()
seattle01_crime_df.head()
seattle02_crime_df.head()


# ## Resources and References
# *What resources and references have you used for this project?*
# I used Kaggle to source the datasets

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


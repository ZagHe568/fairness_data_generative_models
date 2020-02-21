# Data preprocessing
# 1. Add header to data
# 2. Remove unknown: sed '/\?/d' adult.data > adult_known.data
# 3. Remove tailing . in "adult.test": sed 's/.$//' adult_known.test > adult_known.test2

import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
import numpy as np
import scipy
import scipy.optimize
import pdb
import tensorflow as tf
import scipy.stats

tf.random.set_random_seed(2019)

# because we need to encode categorical feature, have to concate dataframe and then split
#For Adult Dataset

df_raw = pd.read_csv('male_adult_dataset', sep=', ', engine='python')

  


from info_gain import info_gain

summm = 0
colnames = list(df_raw.columns.values)
print(colnames)
for i in colnames:
	if(i != 'two_year_recid'):
		ig  = info_gain.info_gain(df_raw['Y'], df_raw[i]) 
		if(ig <0):
			print(ig)
			print(i)
		else:
			worksheet.write(row, 0, i) 
			worksheet.write(row, 1, ig)
			summm = summm +ig
			row = row+1
print(summm)
      
workbook.close() 

ig  = info_gain.info_gain(df_raw['Y'], df_raw['capital-gain'])
print(ig)
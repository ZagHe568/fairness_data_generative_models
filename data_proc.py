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

tf.random.set_random_seed(2019)

# because we need to encode categorical feature, have to concate dataframe and then split
#For Adult Dataset
df_train_raw = pd.read_csv('adult_known.data', sep=', ', engine='python')
df_dev_raw = pd.read_csv('adult_nina_dev', sep=', ', engine='python')
df_test_raw = pd.read_csv('adult_nina_test', sep=', ', engine='python')


n_train = len(df_train_raw)
n_dev = len(df_dev_raw)
n_test = len(df_test_raw)
df_raw = pd.concat([df_train_raw, df_dev_raw,df_test_raw])

#For Adult Dataset
df = pd.get_dummies(df_raw, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'])


# binrary feature will be mapped to two categories. Remove one of them.
#For Adult Dataset
df = df.drop(columns=['sex_Male', 'Y_<=50K'])
X = df.drop(columns=['Y_>50K'])
group_label = X['sex_Female']


scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
#For Adult Dataset
Y = df['Y_>50K']




import tensorflow as tf
import numpy as np
import pandas as pd

from six.moves import urllib
from tensorflow import feature_column


dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(feature_column.numeric_column(feature_name, dtype=tf.float32))

print(dftrain[feature_name].unique())
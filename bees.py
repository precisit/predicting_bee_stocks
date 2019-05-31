import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from bees_stats import *
from bees_encoder import *
from bees_imputation import *
from bees_models import *
import warnings


warnings.filterwarnings("ignore")

print('\n')

data = pd.read_csv('data/data_v3.csv')
stats = check_dataset(data)
'''
# drop columns with risk of data leakage/contamination
col_leak = ['yieldpercol', 'totalprod', 'stocks', 'priceperlb', 'prodvalue']
data_drop = data.copy()
data_drop.drop(col_leak, axis=1, inplace=True)
#plot_corr(data_drop)

# split to train, validation and test sets
# remove rows with missing target, separate target from predictors
data_drop.dropna(axis=0, subset=['numcol'], inplace=True)
y = data_drop.numcol
data_drop.drop(['numcol'], axis=1, inplace=True)

# break off validation set from training data
# FIXME behövs fortf test!
xt, xv, yt, yv = train_test_split( # x training and validation, y training and validation
	data_drop, y, train_size=0.8, test_size=0.2, random_state=0)

# encode categorical variables
# fixme 
xt_label, xv_label = label_encoder(xt, xv) #, yt, yv, print_out=True)

# handle missing values, get imputated xt and xv
xt_i, xv_i = best_imputation(xt_label, xv_label, yt, yv)

# fixme implementera pipelines

# compare performance of decision tree, random forest and gradient boosting
model = best_model(xt_i, xv_i, yt, yv)
print(yv.mean())


'''

print('\n')
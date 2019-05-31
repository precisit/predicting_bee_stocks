from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# comparing different approaches
def score_dataset(xt, xv, yt, yv):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    print(xt)
    model.fit(xt, yt)
    preds = model.predict(xv)
    return mean_absolute_error(yv, preds)

def label_encoder(xt, xv):
	# get categorical columns
	object_cols = [col for col in xt.columns if xt[col].dtype == "object"]

	# check which columns to encode and which to drop
	cols_keep = [col for col in object_cols if set(xt[col]) == set(xv[col])]
	cols_drop = list(set(object_cols)-set(cols_keep))
	xt_label = xt.drop(cols_drop, axis=1)
	xv_label = xv.drop(cols_drop, axis=1)
	        
	#print('Categorical columns to label encod:', cols_keep)
	#print('\n')
	#print('Categorical columns to drop:', cols_drop)

	le = LabelEncoder()
	for col in set(cols_keep):
	    xt_label[col] = le.fit_transform(xt_label[col])
	    xv_label[col] = le.transform(xv_label[col])

	return xt_label, xv_label

def best_encoder(xt, xv, yt, yv, print_out):
	xt_label, xv_label = label_encoder(xt, xv)
	if print_out:
		print('MAE label encoder: ', score_dataset(xt_label, xv_label, yt, yv))
		#print('MAE one-hot encoder: ', score_dataset(xt, xv, yt, yv))
	return xt_label, xv_label

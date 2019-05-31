from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.impute import SimpleImputer
from impyute.imputation.cs import mice
from fancyimpute import KNN 
import sys
import pandas as pd

STRATEGIES = ['mean', 'median', 'most_frequent', 'constant', 'mice', 'knn']
K = 5

sys.setrecursionlimit(100000) # increase recursion limit of os for knn

def mae(xt, xv, yt, yv):
    model = RFR(n_estimators=100, random_state=0)
    model.fit(xt, yt)
    preds = model.predict(xv)
    return mean_absolute_error(yv, preds)

def impute(xt, xv, strategy):
	if strategy == 'mice':	
		xt_imputed = mice(xt)
		xv_imputed = mice(xv)
	elif strategy == 'knn':
		xt_imputed = KNN(k=K, verbose=False).fit_transform(xt)
		xv_imputed = KNN(k=K, verbose=False).fit_transform(xv)
	else:
		imp = SimpleImputer(strategy=strategy)
		xt_imputed = pd.DataFrame(imp.fit_transform(xt))
		xv_imputed = pd.DataFrame(imp.transform(xv))
		# put column names back after imputation
		xt_imputed.columns = xt.columns
		xv_imputed.columns = xv.columns
	return xt_imputed, xv_imputed

def best_imputation(xt, xv, yt, yv):
	# compare imputation with mean, median, most frequent, k-NN and MICE
	xt_i, xv_i = impute(xt, xv, STRATEGIES[0])
	error = mae(xt_i, xv_i, yt, yv)
	for s in STRATEGIES[1:len(STRATEGIES)]:
		xt_s, xv_s = impute(xt, xv, s)
		e = mae(xt_s, xv_s, yt, yv)
		if e < error:
			xt_i = xt_s
			xv_i = xv_s
			error = e
	return xt_i, xv_i

def best_strategy(xt, xv, yt, yv):
	# find best strategy for imputing
	strategy = STRATEGIES[0]
	xt_i, xv_i = impute(xt, xv, strategy)
	error = mae(xt_i, xv_i, yt, yv)
	for s in STRATEGIES[1:len(STRATEGIES)]:
		xt_s, xv_s = impute(xt, xv, s)
		e = mae(xt_s, xv_s, yt, yv)
		if e < error:
			strategy = s
			error = e
	return s

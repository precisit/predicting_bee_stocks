import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
# models
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost import XGBRegressor as xgb

def get_mae(model, x, y): # val x y
	pred = model.predict(x)
	return mae(pred, y)

def best_model(xt, xv, yt, yv):
	models = []

	name_dt = "DecisionTreeRegressor"
	model_dt = dtr(random_state=1) # decision tree
	model_dt.fit(xt, yt)
	models.append({'name': name_dt, 'model': model_dt, 'mae': get_mae(model_dt, xv, yv)})

	name_rf = "RandomForestRegressor"
	model_rf = rfr(random_state=1) # random forest
	model_rf.fit(xt, yt)
	models.append({'name': name_rf, 'model': model_rf, 'mae': get_mae(model_rf, xv, yv)})

	name_xgb = "XGBRegressor"
	model_xgb = xgb(random_state=1, n_estimators=10000, learning_rate=0.01) # xgboost
	model_xgb.fit(xt, yt, early_stopping_rounds=10, eval_set=[(xv, yv)], verbose=False)
	models.append({'name': name_xgb, 'model': model_xgb, 'mae': get_mae(model_xgb, xv, yv)})
	
	print("\n")
	for m in models:
		print("Model {} has MAE {}".format(m.get('name'), m.get('mae')))

	min_mae = min(i['mae'] for i in models)
	best_model = [m for m in models if m.get('mae') == min_mae]
	print("\nBest model pick: ", best_model[0].get('name'))
	print("\n")

	return best_model[0].get('model')
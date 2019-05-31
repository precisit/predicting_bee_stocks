import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def check_dataset(ds):
	'''
	info = input("\nPrint dataset info? [y/n] ").lower()
	if info == 'y':
		print('\n')
		print('Shape: ', ds.shape)
		print('Column names:')
		print(ds.columns.values)
		print('\n')

	columns = input("\nPrint column info? [y/n] ").lower()
	if columns == 'y':
		print('\n')
		print('Types: ', ds.dtypes)
		print('Number of NaNs:')
		print(ds.isnull().sum())
		print('\n')

	description = input("\nPrint dataset desciption? [y/n] ").lower()
	if description == 'y':
		print('\n')
		print(ds.describe())
		print('\n')

	correlation = input("\nPlot dataset correlation? [y/n] ").lower()
	if correlation == 'y':
		print('\n')
		plot_corr(ds)
		print('\n')

	year = input("\nPlot pesticide use per year? [y/n] ").lower()
	if year == 'y':
		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.set_title('Neonicotinoid use per year')
		ax1.plot(ds.groupby('year')['nAllNeonic'].sum())
		ax2.set_title('Bee stocks per year')
		ax2.plot(ds.groupby('year')['stocks'].sum())
		plt.show()
		print('\n')

	per_state = input("\nPlot pesticide use per state? [y/n] ").lower()
	if per_state == 'y':
		states = ds['state'].unique()
		years = np.sort(ds['year'].unique())

		fig, ax = plt.subplots(nrows=11, ncols=4)
		i = 0
		for row in ax:
			for col in row:
				s = ds.loc[ds['state'] == states[i]]
				s = s.set_index('year').sort_values(by='year')
				col.set_title(states[i])
				col.plot(s['nAllNeonic'])
				i += 1

		plt.show()

	colony = input("\nPlot pesticide use and bee colony size for a state? [y/n] ").lower()
	if colony == 'y':
		states = ds['state'].unique()
		df_corr = pd.DataFrame(columns=['Correlation'])
		years = np.sort(ds['year'].unique())
		for state in states:
			s = ds.loc[ds['state'] == state]
			s = s.set_index('year').sort_values(by='year')
			fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
			fig.suptitle(state)
			ax1.set_title('Neonic')
			ax1.plot(s['nAllNeonic'])
			ax2.set_title('Bee stocks')
			ax2.plot(s['stocks'])
			s_corr = s['nAllNeonic'].corr(s['stocks'])
			#print("%s Correlation neonic use and bee stocks:  %0.3f" % (state, s_corr))
			df_corr.loc[state] = s_corr
			plt.style.use(['seaborn-dark'])
			if state == 'WA':
				plt.show()
		print('\n')
		return df_corr

	'''

def plot_corr(ds):
	ds_corr = ds.corr()
	fig = plt.figure()
	subplot = fig.add_subplot(111)
	colors = subplot.matshow(ds_corr, cmap='coolwarm', vmin=-1, vmax=1)
	fig.colorbar(colors)

	ticks = np.arange(0, len(ds.columns), 1)
	subplot.set_xticks(ticks)
	plt.xticks(rotation=90)
	subplot.set_yticks(ticks)
	subplot.set_xticklabels(ds.columns)
	subplot.set_yticklabels(ds.columns)
	plt.show()
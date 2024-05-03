from pandas import HDFStore

store = HDFStore('model.h5')

store['table1Name'].to_csv('Fmodel.xml')
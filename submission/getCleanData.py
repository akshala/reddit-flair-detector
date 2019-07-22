# reference https://stackoverflow.com/questions/42007318/pandas-apply-a-specific-function-to-columns-and-create-other-columns
# reference https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html

import cleaning
import pandas

data = pandas.read_csv('initial_final.csv')

data[['title']] = data.apply(lambda x: cleaning.clean(x['title']), axis=1)
data[['comments']] = data.apply(lambda x: cleaning.clean(x['comments']), axis=1)
data.to_csv('cleaned_data_final.csv', index=False) 

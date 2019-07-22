# reference https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# reference https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import pickle

import warnings
warnings.filterwarnings("ignore")

data = pandas.read_csv('cleaned_data_final.csv')
y = data['link_flair_text']
flairs = list(set(y))

X = data['title'] + data['url'] + data['comments'] + data['author'].fillna('') + data['id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
pickle.dump(logreg, open("log_reg_combined_model.sav", 'wb'))
y_pred = logreg.predict(X_test)
print('accuracy (combined)', accuracy_score(y_pred, y_test))

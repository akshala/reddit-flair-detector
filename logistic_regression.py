# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
# https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings

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

import warnings
warnings.filterwarnings("ignore")

data = pandas.read_csv('cleaned_data_final.csv')
y = data['link_flair_text']
flairs = list(set(y))

X = data['title']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('accuracy (title)', accuracy_score(y_pred, y_test))


X = data['url']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('accuracy (url)', accuracy_score(y_pred, y_test))


X = data['comments']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('accuracy (comments)', accuracy_score(y_pred, y_test))

X = data['author'].fillna('')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('accuracy (author)', accuracy_score(y_pred, y_test))

X = data['id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('accuracy (id)', accuracy_score(y_pred, y_test))
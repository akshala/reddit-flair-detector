# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
# https://stackoverflow.com/questions/22341271/get-list-from-pandas-dataframe-column
# https://stackoverflow.com/questions/52632777/getting-nameerror-name-countvectorizer-is-not-defined-in-pycharm
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# https://github.com/scikit-learn/scikit-learn/issues/13055 - extra
# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi/47285662 - extra
# https://www.oipapio.com/question-819757
# https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
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
nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('accuracy (title)', accuracy_score(y_pred, y_test))


X = data['url']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('accuracy (url)', accuracy_score(y_pred, y_test))


X = data['comments']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('accuracy (comments)', accuracy_score(y_pred, y_test))

X = data['author'].fillna('')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('accuracy (author)', accuracy_score(y_pred, y_test))

X = data['id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
nb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print('accuracy (id)', accuracy_score(y_pred, y_test))


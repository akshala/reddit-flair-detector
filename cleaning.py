# reference https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

import nltk
# nltk.download('all')
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean(text):
	text = BeautifulSoup(text, "lxml").text # HTML decoding
	text = text.lower() # lowercase text
	text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
	text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
	text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
	return text
	
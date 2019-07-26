# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

import praw
import pandas
import cleaning
import pickle

reddit = praw.Reddit(client_id='', client_secret='', user_agent='', username='', password='')
submission = reddit.submission(url="link_you_want_to_check")

posts = []
submission.comments.replace_more(limit=None)
commentList = ''
for comment in submission.comments:
	commentList += " " + comment.body
posts.append([submission.link_flair_text, submission.title, submission.id, submission.url, submission.created, commentList, submission.author])
data = pandas.DataFrame(posts,columns=['link_flair_text', 'title','id', 'url', 'created', 'comments', 'author'])
data[['title']] = data.apply(lambda x: cleaning.clean(x['title']), axis=1)
data[['comments']] = data.apply(lambda x: cleaning.clean(x['comments']), axis=1)

X = data['title'] + data['url'] + data['comments'] + data['id']

loaded_model = pickle.load(open("model.sav", 'rb'))
predicted_flair = loaded_model.predict(X)
print(predicted_flair, submission.link_flair_text)

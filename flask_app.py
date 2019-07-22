# refernce https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
# refernce https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

from flask import Flask, render_template
from flask_restful import Resource, Api
from flask import request
import predictor
import praw
import pandas
import cleaning
import pickle

app = Flask(__name__)
api = Api(app)

@app.route("/")
def getPage():
    return render_template('reddit_flair_detector.html')

@app.route("/getFlair", methods=['GET'])
def getFlair():
	link = request.args.get('link')
	reddit = praw.Reddit(client_id='', client_secret='', user_agent='', username='', password='')
	submission = reddit.submission(url=link)

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

	loaded_model = pickle.load(open("log_reg_combined_model.sav", 'rb'))
	predicted_flair = loaded_model.predict(X)
	return str(predicted_flair)

if __name__ == "__main__":
    app.run()
    
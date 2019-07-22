# reference http://www.storybench.org/how-to-scrape-reddit-with-python/
# reference https://towardsdatascience.com/scraping-reddit-with-praw-76efc1d1e1d9
# reference https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
# reference https://praw.readthedocs.io/en/latest/
# reference https://praw.readthedocs.io/en/latest/tutorials/comments.html
# reference https://www.reddit.com/wiki/search
# reference https://github.com/praw-dev/praw/issues/866
# reference https://www.reddit.com/r/redditdev/comments/3eu705/loading_all_top_level_comments_from_a_specific/
# reference https://www.reddit.com/prefs/apps

import praw
import pandas
from pprint import pprint

reddit = praw.Reddit(client_id='', client_secret='', user_agent='', username='', password='')
subreddit = reddit.subreddit('India')

posts = []
for submission in subreddit.top(limit=100):
	if 'author' and 'comments' in dir(submission):
		try:
			submission.comments.replace_more(limit=None)
			commentList = ''
			for comment in submission.comments:
				commentList += " " + comment.body
			print(commentList)
			posts.append([submission.link_flair_text, submission.title, submission.id, submission.url, submission.created, commentList, submission.author])
		except Exception as e:
			pass
data = pandas.DataFrame(posts,columns=['link_flair_text', 'title','id', 'url', 'created', 'comments', 'author'])
data.to_csv('initial_final.csv', index=False) 

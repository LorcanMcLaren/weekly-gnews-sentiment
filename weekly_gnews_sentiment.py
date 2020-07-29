import requests
import sys
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, median, stdev
from scipy.stats import skew

def format_title(query):
	query = query.replace('%20',' ')
	return query.title()


url_prefix = 'https://news.google.com/search?q='
url_suffix = '%20when%3A7d&hl=en-IE&gl=IE&ceid=IE%3Aen'

query = sys.argv[1]

url = url_prefix + str(query) + url_suffix
print(url)

page = requests.get(url)

soup = BeautifulSoup(page.content,'html.parser')

headlines = soup.find_all('a',class_='DY5T1d')

analyzer = SentimentIntensityAnalyzer()

scores = []
for headline in headlines:
    # print(headline.text, end='\n'*2)
    # print(analyzer.polarity_scores(headline.text)['compound'])
    scores.append(analyzer.polarity_scores(headline.text)['compound'])

# print('LENGTH: ' + str(len(headlines)))

print('MEAN: ', mean(scores))
print('MEDIAN: ', median(scores))
print('STD DEV: ', stdev(scores))
print('SKEWNESS: ', skew(scores))

ax = sns.distplot(scores)
title = format_title(query) + ' Sentiment'
ax.set(xlabel='Polarity Score',ylabel='Frequency',title=title)
plt.show()

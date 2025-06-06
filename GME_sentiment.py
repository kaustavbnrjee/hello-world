"""
@author: KB
BUSADM 797
Comprehensive Exam 2025

Sentiment Measure of Online Articles. GAMESTOP (GME) stock articles from Seeking Alpha
GME sample is the Seeking Alpha's sentiment of GME stock around 2018-2019 when the pumping happened
"""

import pandas as pd
import numpy as np
# pd.set_option('display.max_columns',50)
# pd.set_option('display.width',200)
# pd.set_option('mode.chained_assignment',None)
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
nltk.download('punkt_tab')

datafolder = "/Users/kaustav/Dropbox/BUSADM 797-01/Data"
sapath = f"{datafolder}/GMESample"
safiles = [f for f in os.listdir(sapath) if f.endswith('htm')]

lmpath = f'{datafolder}/LoughranMcDonald_SentimentWordLists_2018.xlsx' ### LM dictionary for pos n negative words, next extract these lists
negatives = set(pd.read_excel(lmpath,sheet_name='Negative', names=['word'],header=None)['word'].str.lower())
positives = set(pd.read_excel(lmpath,sheet_name='Positive', names=['word'],header=None)['word'].str.lower())
stopwords = set([w.lower() for w in stopwords.words('english')])

### parse one element, see the HTML in a structured format, this help you to navigate and manipulate the document’s elements, as shown below.
art = '4161066.htm'
with open(f'{sapath}/{art}','r', encoding='utf-8', errors='ignore') as f:
    contents = f.read()

print(contents)
soup = BeautifulSoup(contents,'lxml')
# from each article collect title, firm, timestamp, author, article

record = dict()
soup.find('h1',attrs={'data-test-id':'post-title'}) ### title of the article

ticker = soup.find('span',attrs={'data-test-id':'post-primary-tickers'})
if ticker:
    record['ticker'] = ticker.get_text() ### ticker

timestamp = soup.find('span',attrs={'data-test-id':'post-date'})
if timestamp:
    record['timestamp'] = timestamp.get_text() ### timestamp

author = soup.find('a',attrs={'data-test-id':'author-name'})
if author:
    record['author'] = author.get_text() ### author

article = soup.find('div',attrs={'data-test-id':'article-content'})
if article:
    text = article.get_text() ### article contents

words = word_tokenize(text)   ### Tokenization ### list all the words and its length
#Tokenization is the process of splitting a sequence of text into smaller pieces, such as words or phrases

wordcounts = Counter([w.lower() for w in words if w.isalpha() and w not in stopwords]) ### count frequency of each words in the article

# set up our sentiment variables
record['totalwords'] = sum(wordcounts.values())
record['poswords'] = sum([v for k,v in wordcounts.items() if k in positives])
record['negwords'] = sum([v for k,v in wordcounts.items() if k in negatives])

"""
function parse_article(art)
this defines a  that parses an article from an HTML file and
extracts key information like title, ticker, timestamp, author, and the article's content.
Check number of pos n neg words in the article against LM dictionary of pos and neg words.
"""
def parse_article(art):
    with open(f'{sapath}/{art}', 'r',encoding='utf-8', errors='ignore') as f:
        contents = f.read()
    soup = BeautifulSoup(contents, 'lxml') # create the soup
    record = dict() # set up dictionary to collect info
    # title
    title = soup.find('h1', attrs={'data-test-id': 'post-title'})
    if title:
        record['title'] = title.get_text().strip()
    # ticker
    ticker = soup.find('span', attrs={'data-test-id': 'post-primary-tickers'})
    if ticker:
        record['ticker'] = ticker.get_text()
    # timestamp
    timestamp = soup.find('span', attrs={'data-test-id': 'post-date'})
    if timestamp:
        record['timestamp'] = timestamp.get_text()
    # author
    author = soup.find('a', attrs={'data-test-id': 'author-name'})
    if author:
        record['author'] = author.get_text()
    # article contents
    article = soup.find('div', attrs={'data-test-id': 'article-content'})
    text = ""
    if article:
        text = article.get_text()

    if len(text)>0:
        # Compute sentiment
        words = word_tokenize(text)
        wordcounts = Counter([w.lower() for w in words if w.isalpha() and w not in stopwords])
        # set up our sentiment variables
        record['totalwords'] = sum(wordcounts.values())
        record['poswords'] = sum([v for k, v in wordcounts.items() if k in positives])
        record['negwords'] = sum([v for k, v in wordcounts.items() if k in negatives])
    return record

# w.isalpha(): only keep alphabetic words
# w.lower() covert all words to lower for consistency
# stopwrods: e.g. the, is,  and are ignored
# summing the counts (v) of words (k) in wordcounts that are present in the positives

# Enumerate to record all of the GameStop articles
allsa = []
file_names = []
for i,art in enumerate(safiles):
    allsa.append(parse_article(art))
    print(i, art)
    file_names.append(art)

sadf = pd.DataFrame(allsa)
sadf['file'] = file_names

# Simple calulation of sentiment 
sadf['sentiment'] = sadf.eval('(poswords-negwords)/totalwords')
sadf = sadf.dropna(subset='sentiment')
sadf.to_csv('kb_GME_sentiment.csv')
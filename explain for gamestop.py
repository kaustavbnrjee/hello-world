#!/usr/bin/env python
# coding: utf-8

# In[3]:


#############################################
####MEASURE SENTIMENT OF ONLINE ARTICLE######
#############################################



import pandas as pd, numpy as np
pd.set_option('display.max_columns',50)
pd.set_option('display.width',200)
pd.set_option('mode.chained_assignment',None)

datafolder =  "/Users/mohammadhosseinrashidi/Dropbox/BUSADM 797-01/Data"


# Type 4: Seeking Alpha Articles (measure sentiment)
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
nltk.download('punkt_tab')

##############
###GME sample is the Seeking Alpha's sentiment of GME stock around 2018-2019 when the pumping happened


sapath = f"{datafolder}/GMESample"
safiles = [f for f in os.listdir(sapath) if f.endswith('htm')]

# from os import scandir
# for f in scandir(sapath):
#     print(f.name)
#     print(f.path)
#     break

lmpath = f'{datafolder}/LoughranMcDonald_SentimentWordLists_2018.xlsx'     ###take the positive and negative word from the dictionary

####Next: extract the positive and negative word from the dictionary

negatives = set(pd.read_excel(lmpath,sheet_name='Negative',
                              names=['word'],header=None)['word'].str.lower())



positives = set(pd.read_excel(lmpath,sheet_name='Positive',
                              names=['word'],header=None)['word'].str.lower())

stopwords = set([w.lower() for w in stopwords.words('english')])







#####take a sample from the link to test the function later #########


###This line opens the file located at the path f'{sapath}/{art}' (constructed by concatenating the sapath directory and the file name art) in read mode ('r').

art = '4161066.htm'
with open(f'{sapath}/{art}','r', encoding='utf-8', errors='ignore') as f:
    contents = f.read()
    

#parsing the document, represents the HTML document in a structured format, allowing you to easily navigate and manipulate the document’s elements 

soup = BeautifulSoup(contents,'lxml')

# from each article collect title, firm, timestamp, author, article
record = dict()
# title
soup.find('h1',attrs={'data-test-id':'post-title'})




    
    
    
# ticker
ticker = soup.find('span',attrs={'data-test-id':'post-primary-tickers'})
if ticker:
    record['ticker'] = ticker.get_text()
    
    
    
# timestamp
timestamp = soup.find('span',attrs={'data-test-id':'post-date'})
if timestamp:
    record['timestamp'] = timestamp.get_text()
    
    
# author
author = soup.find('a',attrs={'data-test-id':'author-name'})
if author:
    record['author'] = author.get_text()
    
    
    
# article contents
article = soup.find('div',attrs={'data-test-id':'article-content'})
if article:
    text = article.get_text()
    


#Tokenization is the process of splitting a sequence of text into smaller pieces, such as words or phrases
words = word_tokenize(text)   #list all the words and its length

#count frequency of each words in the article
wordcounts = Counter([w.lower() for w in words if w.isalpha() and w not in stopwords])













# set up our sentiment variables
record['totalwords'] = sum(wordcounts.values())
record['poswords'] = sum([v for k,v in wordcounts.items() if k in positives])
record['negwords'] = sum([v for k,v in wordcounts.items() if k in negatives])


#This code defines a function parse_article(art) that parses an article 
#from an HTML file and extracts key information like title, ticker,
# timestamp, author, and the article's content.





#The total number of words, positive words, and negative words is 
#computed by comparing each word in the text to predefined lists of 
#positive (positives) and negative (negatives) words.



def parse_article(art):
    with open(f'{sapath}/{art}', 'r',encoding='utf-8', errors='ignore') as f:
        contents = f.read()
    # Create the soup
    soup = BeautifulSoup(contents, 'lxml')
    # set up dictionary to collect info
    record = dict()
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


#w.isalpha(): only keep alphabetic words
#w.lower() covert all words to lower for consistency
#stopwrods: e.g. the, is,  and are ignored
#summing the counts (v) of words (k) in wordcounts that are present in the positives






#Enumerate to record all of the GameStop articles

allsa = []
for i,art in enumerate(safiles):
    allsa.append(parse_article(art))
    print(i)
    
    



sadf = pd.DataFrame(allsa)

#Simple calulation of sentiment 
sadf['sentiment'] = sadf.eval('(poswords-negwords)/totalwords')


# In[ ]:


# Load essential libraries for data handling
import pandas as pd, numpy as np  # Data analysis libraries
# کتابخانه‌های پایتون برای تحلیل داده

# Configure pandas to display more columns and avoid warnings
pd.set_option('display.max_columns',50)  # Show up to 50 columns
pd.set_option('display.width',200)       # Set terminal width
pd.set_option('mode.chained_assignment',None)  # Suppress SettingWithCopyWarning
# تنظیمات برای بهتر دیدن داده‌ها و جلوگیری از اخطارهای غیر ضروری

# Set the data folder path
datafolder =  "/Users/mohammadhosseinrashidi/Dropbox/BUSADM 797-01/Data"
# مسیر فولدر حاوی داده‌ها را مشخص می‌کنیم

# Load sentiment analysis libraries
from bs4 import BeautifulSoup  # For parsing HTML
import os  # To interact with the operating system (file reading)
from nltk.tokenize import word_tokenize  # For splitting text into words
from nltk.corpus import stopwords  # Common English stop words
from collections import Counter  # Count frequency of words
import nltk
nltk.download('punkt_tab')  # Ensure tokenization resources are downloaded
# کتابخانه‌های لازم برای تجزیه و تحلیل متن و خواندن فایل‌های HTML

# Define the path for Seeking Alpha articles related to GME
sapath = f"{datafolder}/GMESample"
safiles = [f for f in os.listdir(sapath) if f.endswith('htm')]  # Filter HTML files
# مسیر مقالات سایت Seeking Alpha مربوط به سهام GME در سال‌های خاص

# Define path for Loughran and McDonald sentiment dictionary
lmpath = f'{datafolder}/LoughranMcDonald_SentimentWordLists_2018.xlsx' 
# مسیر دیکشنری واژگان مثبت و منفی اقتصادی

# Load negative words
negatives = set(pd.read_excel(lmpath,sheet_name='Negative',
                              names=['word'],header=None)['word'].str.lower())
# بارگذاری لیست کلمات منفی از فایل اکسل و تبدیل به مجموعه (set)

# Load positive words
positives = set(pd.read_excel(lmpath,sheet_name='Positive',
                              names=['word'],header=None)['word'].str.lower())
# بارگذاری لیست کلمات مثبت از فایل اکسل و تبدیل به مجموعه (set)

# Get standard English stopwords (like "the", "is", etc.)
stopwords = set([w.lower() for w in stopwords.words('english')])
# بارگذاری و کوچک‌نویسی کلمات رایج بی‌اهمیت در زبان انگلیسی

# Choose a sample article for testing
art = '4161066.htm'

# Read HTML content from the file
with open(f'{sapath}/{art}','r', encoding='utf-8', errors='ignore') as f:
    contents = f.read()
# خواندن محتوای یک مقاله نمونه

# Parse HTML using BeautifulSoup
soup = BeautifulSoup(contents,'lxml')
# تجزیه فایل HTML برای استخراج عناصر مهم

record = dict()  # Dictionary to store parsed info

# Extract ticker
ticker = soup.find('span',attrs={'data-test-id':'post-primary-tickers'})
if ticker:
    record['ticker'] = ticker.get_text()
# استخراج تیکر شرکت (مثلاً GME)

# Extract timestamp
timestamp = soup.find('span',attrs={'data-test-id':'post-date'})
if timestamp:
    record['timestamp'] = timestamp.get_text()
# استخراج زمان انتشار مقاله

# Extract author
author = soup.find('a',attrs={'data-test-id':'author-name'})
if author:
    record['author'] = author.get_text()
# استخراج نام نویسنده مقاله

# Extract full article content
article = soup.find('div',attrs={'data-test-id':'article-content'})
if article:
    text = article.get_text()
# استخراج محتوای کامل مقاله

# Tokenize words and count word frequency (ignore punctuation and stopwords)
words = word_tokenize(text)
wordcounts = Counter([w.lower() for w in words if w.isalpha() and w not in stopwords])
# توکن‌سازی متن، حذف علائم نگارشی و کلمات رایج، شمارش تعداد دفعات هر کلمه

# Compute sentiment metrics
record['totalwords'] = sum(wordcounts.values())
record['poswords'] = sum([v for k,v in wordcounts.items() if k in positives])
record['negwords'] = sum([v for k,v in wordcounts.items() if k in negatives])
# محاسبه تعداد کل کلمات، کلمات مثبت و منفی

# Define the main function to extract and process an article
def parse_article(art):
    with open(f'{sapath}/{art}', 'r',encoding='utf-8', errors='ignore') as f:
        contents = f.read()
    soup = BeautifulSoup(contents, 'lxml')
    record = dict()

    title = soup.find('h1', attrs={'data-test-id': 'post-title'})
    if title:
        record['title'] = title.get_text().strip()

    ticker = soup.find('span', attrs={'data-test-id': 'post-primary-tickers'})
    if ticker:
        record['ticker'] = ticker.get_text()

    timestamp = soup.find('span', attrs={'data-test-id': 'post-date'})
    if timestamp:
        record['timestamp'] = timestamp.get_text()

    author = soup.find('a', attrs={'data-test-id': 'author-name'})
    if author:
        record['author'] = author.get_text()

    article = soup.find('div', attrs={'data-test-id': 'article-content'})
    text = ""
    if article:
        text = article.get_text()

    if len(text)>0:
        words = word_tokenize(text)
        wordcounts = Counter([w.lower() for w in words if w.isalpha() and w not in stopwords])
        record['totalwords'] = sum(wordcounts.values())
        record['poswords'] = sum([v for k, v in wordcounts.items() if k in positives])
        record['negwords'] = sum([v for k, v in wordcounts.items() if k in negatives])

    return record
# تعریف تابع اصلی برای استخراج اطلاعات کامل از یک مقاله و محاسبه سنجش احساس

# Loop through all Seeking Alpha articles and parse them
allsa = []
for i,art in enumerate(safiles):
    allsa.append(parse_article(art))
    print(i)
# اجرای تابع روی تمام مقالات برای استخراج احساس کلی

# Store results in a DataFrame
sadf = pd.DataFrame(allsa)
# تبدیل لیست دیکشنری‌ها به دیتافریم

# Calculate sentiment score
sadf['sentiment'] = sadf.eval('(poswords-negwords)/totalwords')
# محاسبه نمره احساس با فرمول ساده: مثبت منفی تقسیم بر کل کلمات


# # 🟦 Scenario 1: Add Neutral Words
# Request: “I want to include neutral words in the analysis.”
# 
# What to change:
# 
# Add a new sheet to the Loughran-McDonald Excel for neutral words.
# 
# Load them just like positives and negatives:
# 
# #### start code:
# neutrals = set(pd.read_excel(lmpath, sheet_name='Neutral',
#                              names=['word'], header=None)['word'].str.lower())
# ### continueAdd this to parse_article to calculate neutral word count:
# 
# 
# record['neutralwords'] = sum([v for k, v in wordcounts.items() if k in neutrals])
# ### finish
# فارسی: برای لحاظ کردن کلمات خنثی، یک Sheet جدید به فایل اکسل اضافه کن و مشابه کلمات مثبت و منفی آن را بارگذاری کن.
# 
# 

# # 🟦 Scenario 2: Filter Articles by Date Range
# Request: “Only analyze articles from Jan 2019 to Dec 2019.”
# 
# What to do:
# 
# Inside the parse_article loop, convert timestamp to datetime:
# 
# ### start
# record['timestamp'] = pd.to_datetime(record['timestamp'])
# ### continue Filter sadf after creation:
# 
# 
# sadf = sadf[(sadf['timestamp'] >= '2019-01-01') & (sadf['timestamp'] <= '2019-12-31')]
# ### finish
# فارسی: تبدیل تاریخ به datetime و سپس فیلتر کردن فقط مقاله‌هایی که در بازه ۲۰۱۹ قرار دارند.
# 
# 

# # Scenario 3: Add Sentiment Categories (Positive/Neutral/Negative)
# Request: “Label each article as ‘Positive’, ‘Neutral’, or ‘Negative’.”
# 
# What to add:
# 
# ### start
# def label_sentiment(score):
#     if score > 0.01:
#         return 'Positive'
#     elif score < -0.01:
#         return 'Negative'
#     else:
#         return 'Neutral'
# 
# sadf['sentiment_label'] = sadf['sentiment'].apply(label_sentiment)
# ### finish
# فارسی: یک تابع تعریف کن که بر اساس امتیاز احساس، برچسب مثبت، منفی یا خنثی بدهد.

# # Scenario 4: Use More Advanced Sentiment Models (e.g., VADER)
# Request: “Can we use a machine learning-based or pretrained model like VADER?”
# 
# What to do:
# 
# Add from nltk.sentiment.vader import SentimentIntensityAnalyzer
# 
# Load analyzer: sia = SentimentIntensityAnalyzer()
# 
# In parse_article, replace simple counts:
# 
# ### start
# sentiment_scores = sia.polarity_scores(text)
# record['compound_score'] = sentiment_scores['compound']
# ### fnish
# فارسی: به جای شمارش دستی کلمات مثبت و منفی، از مدل آماده NLTK مثل VADER استفاده کن که دقت بیشتری داره.
# 
# 

# # 5. Track Article Length as a Feature
# Question: “How would you measure the average article length and assess its relationship with sentiment?”
# 
# Answer:
# 
# Add in parse_article():
# 
# ### start
# record['article_length'] = len(text.split())  # Word count of article
# ### Reason:
# 
# Helps test if longer articles tend to be more positive/negative.
# 
# Enables regression or visualization.
# 
# 6. Normalize Sentiment by Paragraph or Sentence
# Question: “What if longer articles distort sentiment scores?”
# 
# Answer:
# 
# Use average sentiment per sentence:
# 
# ### start
# 
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt')  # Required for sentence splitting
# record['avg_sentiment'] = sentiment_scores['compound'] / len(sent_tokenize(text))
# ### Reason:
# 
# Normalization prevents longer articles from biasing results.
# 
# 7. Store Raw Article Text for Later Use
# Question: “How can we reuse text for LLM analysis or vector embedding?”
# 
# Answer:
# 
# Add to record:
# 
# ### start
# 
# record['raw_text'] = text
# ### Reason:
# 
# Needed for text classification, GPT summarization, etc.
# 
# 8. Handle Missing or Corrupt Articles
# Question: “Some files are broken or empty. What’s your solution?”
# 
# Answer:
# 
# Wrap code in try–except:
# 
# ### start
# 
# try:
#     with open(...) as f:
#         ...
# except Exception as e:
#     print(f"Skipping {art}: {e}")
#     return None
# In the loop:
# 
# ### start
# 
# parsed = parse_article(art)
# if parsed:
#     allsa.append(parsed)
# ### Reason:
# 
# Makes pipeline robust and production-ready.
# 
# 9. Compare LM Sentiment vs. VADER
# Question: “How can you test if LM and VADER agree?”
# 
# Answer:
# 
# Calculate correlation:
# 
# ### start
# 
# sadf[['sentiment', 'compound_score']].corr()
# ### Reason:
# 
# Validate consistency across different methods.
# 
# 10. Export Results to Excel/CSV
# Question: “How can you save sentiment scores for future use?”
# 
# Answer:
# 
# ### start
# 
# sadf.to_csv('gme_sentiment.csv', index=False)
# ### Reason:
# 
# Needed for sharing, modeling, dashboarding.
# 
# 11. Add Ticker Filtering
# Question: “What if I only want articles about a specific ticker, like 'GME'?”
# 
# Answer:
# 
# ### start
# 
# sadf = sadf[sadf['ticker'] == 'GME']
# ### Reason:
# 
# Allows comparison across firms or sectors.
# 
# 12. Time Series Analysis of Sentiment
# Question: “Can we track how sentiment changed over time?”
# 
# Answer:
# 
# ### start
# 
# sadf.set_index('timestamp').resample('M')['sentiment'].mean().plot()
# ### Reason:
# 
# Useful for matching sentiment to stock price or events.
# 
# 13. Detect Sarcasm or Complex Tone
# Advanced Bonus Question:
# “Can this pipeline detect sarcasm?”
# 
# Answer:
# 
# No, rule-based methods like LM or VADER don’t handle sarcasm.
# 
# Need fine-tuned transformer models (e.g., BERT, GPT).
# 
# 14. Visualize Top Positive/Negative Words
# Question: “Can you visualize the most common sentiment words?”
# 
# Answer:
# 
# ### start
# 
# from wordcloud import WordCloud
# WordCloud().generate_from_frequencies({k:v for k,v in wordcounts.items() if k in positives}).to_image()

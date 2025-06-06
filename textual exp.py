#!/usr/bin/env python
# coding: utf-8

# In[3]:



get_ipython().system('pip install ftfy')
get_ipython().system('pip install sklearn')


# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:19:25 2024

@author: tranm
"""

### Exercise 1 ###


import nltk

import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
import os, pandas as pd, emoji, re, numpy as np

pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',200)

datafolder = "/Users/mohammadhosseinrashidi/Dropbox/BUSADM 797-01/Data"
stpath = f'{datafolder}/GMEstocktwitsV2_sample.csv.gz'

stdf = pd.read_csv(stpath)

sample = stdf.sample(1000,random_state=7)
# get sample tweet:  #not important, just a test
test = sample['text'].tolist()[0]
Counter(word_tokenize(test))


#Create a new column wich is a bag of words
#sample['bow'] = sample['text'].apply(lambda x: Counter(word_tokenize(str(x))) if x is not None else Counter())





#dtm1 = pd.DataFrame(sample['bow'].values.tolist()).fillna(0)
# dtm1 not very good, not the best way to generate


# Replace np.nan with an empty string and convert all entries to string
sample['text'] = sample['text'].fillna('').astype(str)



vec = CountVectorizer()  # Initializes a word count vectorizer
dtm2 = vec.fit_transform(sample['text']) #Transforms text data into a sparse document-term matrix.




dtm2.todense() # generates numpy matrix
#np.asarray(dtm2.todense()) # generates numpy array


example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)     #Creates a DataFrame of word counts with proper labels for columns (words) and rows (documents).

#.todense() generates a matrix object, which behaves differently from
# a regular ndarray. If you need the flexibility of working with a 
#general ndarray (which supports more operations and has better 
#compatibility with most NumPy functions), you convert it using np.asarray()



dtm2.sum(axis=1) # Total word counts per document.
dtm2.sum(axis=0) # Total counts of each word across all documents.






example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)     #Creates a DataFrame of word counts with proper labels for columns (words) and rows (documents).


example.sum(axis=1) # Total word counts per document.
example.sum(axis=0)

#axis=0 means summing across rows
#axis=1 means summing across columns 


###ANOTHER WAY TO DO THIS

import ftfy

#The ftfy library is used here to fix encoding errors and clean up
# text data. Specifically, ftfy.ftfy is a function that attempts 
#to "fix" common problems with text, such as garbled characters, 
#invalid Unicode sequences, or other encoding issues.


#a = ftfy.ftfy('ªð')


sample['text'] = sample['text'].apply(ftfy.ftfy)   # Cleans text data by fixing encoding issues using the ftfy library




dtm2 = vec.fit_transform(sample['text'])  #Converts the cleaned text into a document-term matrix of word counts using CountVectorizer



dtm2_df = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)

#Converts the sparse matrix of word counts into a dense matrix and 
#creates a DataFrame, labeling the columns with the words (features)
#and keeping the original document indices.


dtm2_df.sum(axis=0).sort_values(ascending=False)   # Total counts of each word across all documents.


dtm2_df.sum(axis=1).sort_values(ascending=False)  # Total word counts per document.



#create another dataset without stopwords

from nltk.corpus import stopwords
stops = stopwords.words('english')   
vec = CountVectorizer(stop_words=stops,min_df=2)     

#This argument passes the stops list (which contains the English stopwords) to the CountVectorizer. As a result, any word in the stopwords list will be ignored or filtered out when the CountVectorizer processes the text

#  min_df=2: only words that appear in at least 2 documents will be considered. Words that appear in fewer than 2 documents will be excluded from the document-term matrix.

dtm2b = vec.fit_transform(sample['text'])
dtm2b_df = pd.DataFrame(dtm2b.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)


dtm2b_df.sum(axis=0).sort_values(ascending=False) 

dtm2b_df.sum(axis=1).sort_values(ascending=False)   # Total counts of each word across all documents.






# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:19:25 2024

@author: tranm
"""

### Exercise 1 ###


import nltk

import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
import os, pandas as pd, emoji, re, numpy as np

pd.set_option('mode.chained_assignment',None)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',200)

datafolder = "/Users/tranm/Dropbox/UMASS Class/BUSADM 797/Data"
stpath = f'{datafolder}/GMEstocktwitsV2_sample.csv.gz'

stdf = pd.read_csv(stpath)

sample = stdf.sample(1000,random_state=7)
# get sample tweet:  #not important, just a test
test = sample['text'].tolist()[0]
Counter(word_tokenize(test))


#Create a new column wich is a bag of words
#sample['bow'] = sample['text'].apply(lambda x: Counter(word_tokenize(str(x))) if x is not None else Counter())





#dtm1 = pd.DataFrame(sample['bow'].values.tolist()).fillna(0)
# dtm1 not very good, not the best way to generate


# Replace np.nan with an empty string and convert all entries to string
sample['text'] = sample['text'].fillna('').astype(str)



vec = CountVectorizer()  # Initializes a word count vectorizer
dtm2 = vec.fit_transform(sample['text']) #Transforms text data into a sparse document-term matrix.




dtm2.todense() # generates numpy matrix
#np.asarray(dtm2.todense()) # generates numpy array


example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)     #Creates a DataFrame of word counts with proper labels for columns (words) and rows (documents).

#.todense() generates a matrix object, which behaves differently from
# a regular ndarray. If you need the flexibility of working with a 
#general ndarray (which supports more operations and has better 
#compatibility with most NumPy functions), you convert it using np.asarray()



dtm2.sum(axis=1) # Total word counts per document.
dtm2.sum(axis=0) # Total counts of each word across all documents.






example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)     #Creates a DataFrame of word counts with proper labels for columns (words) and rows (documents).


example.sum(axis=1) # Total word counts per document.
example.sum(axis=0)

#axis=0 means summing across rows
#axis=1 means summing across columns 


###ANOTHER WAY TO DO THIS

import ftfy

#The ftfy library is used here to fix encoding errors and clean up
# text data. Specifically, ftfy.ftfy is a function that attempts 
#to "fix" common problems with text, such as garbled characters, 
#invalid Unicode sequences, or other encoding issues.


#a = ftfy.ftfy('ªð')


sample['text'] = sample['text'].apply(ftfy.ftfy)   # Cleans text data by fixing encoding issues using the ftfy library




dtm2 = vec.fit_transform(sample['text'])  #Converts the cleaned text into a document-term matrix of word counts using CountVectorizer



dtm2_df = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)

#Converts the sparse matrix of word counts into a dense matrix and 
#creates a DataFrame, labeling the columns with the words (features)
#and keeping the original document indices.


dtm2_df.sum(axis=0).sort_values(ascending=False)   # Total counts of each word across all documents.


dtm2_df.sum(axis=1).sort_values(ascending=False)  # Total word counts per document.



#create another dataset without stopwords

from nltk.corpus import stopwords
stops = stopwords.words('english')   
vec = CountVectorizer(stop_words=stops,min_df=2)     

#This argument passes the stops list (which contains the English stopwords) to the CountVectorizer. As a result, any word in the stopwords list will be ignored or filtered out when the CountVectorizer processes the text

#  min_df=2: only words that appear in at least 2 documents will be considered. Words that appear in fewer than 2 documents will be excluded from the document-term matrix.

dtm2b = vec.fit_transform(sample['text'])
dtm2b_df = pd.DataFrame(dtm2b.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)


dtm2b_df.sum(axis=0).sort_values(ascending=False) 

dtm2b_df.sum(axis=1).sort_values(ascending=False)   # Total counts of each word across all documents.







# In[ ]:


# -*- coding: utf-8 -*-
# EN: Encoding declaration for UTF-8 to support non-English characters
# FA: تعریف نوع کدگذاری فایل برای پشتیبانی از نویسه‌های غیر انگلیسی

"""
Created on Fri Oct 18 11:19:25 2024
@author: tranm
"""
# EN: Metadata about the file and author
# FA: اطلاعات متا در مورد فایل و نویسنده

### Exercise 1 ###
# EN: Marks the beginning of Exercise 1
# FA: شروع تمرین شماره ۱

import nltk
# EN: Natural Language Toolkit library for tokenizing, stopwords, etc.
# FA: کتابخانه‌ای برای پردازش زبان طبیعی شامل توکن‌سازی و کلمات توقف

import sklearn
# EN: Machine learning library, used here for text vectorization
# FA: کتابخانه یادگیری ماشین که برای تبدیل متن به عدد استفاده می‌شود

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# EN: Tools to transform text into count or tf-idf matrix
# FA: ابزارهایی برای تبدیل متن به ماتریس فرکانس یا TF-IDF

from nltk.tokenize import word_tokenize
# EN: Function to split sentences into words
# FA: توابع برای تقسیم متن به کلمات

from collections import Counter
# EN: For counting frequency of each word
# FA: شمارش تعداد وقوع هر کلمه

import os, pandas as pd, emoji, re, numpy as np
# EN: Load standard libraries for file ops, dataframes, regex, arrays, emoji handling
# FA: بارگذاری کتابخانه‌های عمومی برای کار با فایل، داده‌ها، رشته‌های منظم و ایموجی

pd.set_option('mode.chained_assignment',None)
# EN: Suppresses SettingWithCopyWarning in Pandas
# FA: جلوگیری از هشدارهای مربوط به انتساب زنجیره‌ای در پانداس

pd.set_option('display.max_columns',50)
# EN: Show up to 50 columns in dataframe output
# FA: نمایش حداکثر ۵۰ ستون در خروجی داده

pd.set_option('display.width',200)
# EN: Extend console width for better DataFrame visibility
# FA: تنظیم عرض نمایش برای بهتر دیده شدن داده‌ها

datafolder = "/Users/mohammadhosseinrashidi/Dropbox/BUSADM 797-01/Data"
# EN: Path to the directory containing data
# FA: مسیر فولدر داده‌ها

stpath = f'{datafolder}/GMEstocktwitsV2_sample.csv.gz'
# EN: Full path to the StockTwits dataset file (compressed CSV)
# FA: مسیر فایل فشرده حاوی داده‌های توییتر در مورد GME

stdf = pd.read_csv(stpath)
# EN: Read CSV file into a DataFrame
# FA: خواندن فایل CSV به صورت دیتافریم

sample = stdf.sample(1000,random_state=7)
# EN: Randomly sample 1000 tweets for analysis
# FA: نمونه‌گیری تصادفی از ۱۰۰۰ توییت برای تحلیل

test = sample['text'].tolist()[0]
# EN: Select first tweet from sample for testing
# FA: انتخاب اولین توییت برای آزمایش

Counter(word_tokenize(test))
# EN: Tokenize the tweet and count each word's frequency
# FA: توکن‌سازی توییت و شمارش تعداد تکرار هر کلمه

# sample['bow'] = sample['text'].apply(lambda x: Counter(word_tokenize(str(x))) if x is not None else Counter())
# EN: (Commented) Attempt to build bag-of-words column by word frequency per tweet
# FA: (غیرفعال) ساخت ستون کیسه واژگان برای هر توییت

# dtm1 = pd.DataFrame(sample['bow'].values.tolist()).fillna(0)
# EN: (Commented) Transform bag-of-words dictionary into DataFrame
# FA: (غیرفعال) تبدیل واژه‌نامه به دیتافریم

sample['text'] = sample['text'].fillna('').astype(str)
# EN: Replace missing texts with empty strings and convert all to string type
# FA: پر کردن مقادیر خالی و تبدیل همه به رشته

vec = CountVectorizer()
# EN: Initialize CountVectorizer for creating DTM
# FA: مقداردهی اولیه CountVectorizer برای ساخت ماتریس سند-واژه

dtm2 = vec.fit_transform(sample['text'])
# EN: Fit and transform text data into a sparse DTM
# FA: اعمال CountVectorizer روی داده‌ها برای تولید ماتریس سند-واژه

dtm2.todense()
# EN: Convert sparse matrix to dense matrix (for better visualization)
# FA: تبدیل ماتریس تنک به ماتریس متراکم

example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)
# EN: Create labeled DataFrame of word counts with words as columns
# FA: ساخت دیتافریم با واژه‌ها به عنوان ستون

dtm2.sum(axis=1)
# EN: Total word count per document (tweet)
# FA: تعداد کل کلمات در هر توییت

dtm2.sum(axis=0)
# EN: Total frequency of each word across all documents
# FA: تعداد کل تکرار هر واژه در تمام توییت‌ها

example.sum(axis=1)
# EN: Confirm row sums (word counts per tweet)
# FA: بررسی مجموع ردیف‌ها (تعداد کلمات هر توییت)

example.sum(axis=0)
# EN: Confirm column sums (frequency of each word)
# FA: بررسی مجموع ستون‌ها (تعداد کل یک واژه)

# axis=0 means summing across rows
# axis=1 means summing across columns
# FA: axis=0 یعنی جمع زدن در بین ردیف‌ها و axis=1 یعنی جمع بین ستون‌ها

import ftfy
# EN: Fix Text for You – a library to repair Unicode and text encoding issues
# FA: کتابخانه‌ای برای رفع مشکلات رمزگذاری کاراکترها

sample['text'] = sample['text'].apply(ftfy.ftfy)
# EN: Apply text fixing to clean encoding issues in tweets
# FA: اصلاح مشکلات نویسه‌ها در متن توییت‌ها

dtm2 = vec.fit_transform(sample['text'])
# EN: Re-vectorize cleaned text into document-term matrix
# FA: اعمال CountVectorizer روی متن تمیز شده

dtm2_df = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)
# EN: Convert matrix to DataFrame for word count inspection
# FA: تبدیل ماتریس به دیتافریم برای بررسی فراوانی کلمات

dtm2_df.sum(axis=0).sort_values(ascending=False)
# EN: Sort and view most frequent words in corpus
# FA: مرتب‌سازی و مشاهده پرتکرارترین واژه‌ها

dtm2_df.sum(axis=1).sort_values(ascending=False)
# EN: View tweets with highest word counts
# FA: مشاهده توییت‌هایی با بیشترین تعداد کلمه

from nltk.corpus import stopwords
# EN: Import standard list of English stopwords
# FA: بارگذاری لیست کلمات توقف زبان انگلیسی

stops = stopwords.words('english')
# EN: Load stopwords into a list
# FA: تبدیل کلمات توقف به یک لیست

vec = CountVectorizer(stop_words=stops,min_df=2)
# EN: Reinitialize CountVectorizer to ignore stopwords and exclude rare words
# FA: حذف کلمات پرتکرار و حذف کلمات خیلی کم‌تکرار (کمتر از ۲ سند)

dtm2b = vec.fit_transform(sample['text'])
# EN: Fit and transform text again using new filtering rules
# FA: تولید ماتریس سند-واژه با تنظیمات جدید

dtm2b_df = pd.DataFrame(dtm2b.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)
# EN: Convert matrix to labeled DataFrame
# FA: تبدیل ماتریس به دیتافریم با لیبل‌های واژه

dtm2b_df.sum(axis=0).sort_values(ascending=False)
# EN: Sort most common non-stopwords across tweets
# FA: نمایش پرتکرارترین کلمات بدون کلمات توقف

dtm2b_df.sum(axis=1).sort_values(ascending=False)
# EN: Show tweet lengths based on filtered vocabulary
# FA: نمایش طول توییت‌ها پس از فیلتر شدن واژگان


# In[3]:


🔹 1. Remove Emojis or Special Characters
get_ipython().set_next_input('🧪 Question: How can you clean emojis or special Unicode characters from the tweets');get_ipython().run_line_magic('pinfo', 'tweets')

💡 Why? Emojis are non-standard symbols that can mislead word counts and DTM representations.

✅ Solution:

python
Copy
Edit
import re
sample['text'] = sample['text'].apply(lambda x: re.sub(emoji.get_emoji_regexp(), '', x))


🔹 2. Use TF-IDF Instead of Raw Counts
get_ipython().set_next_input('🧪 Question: Can we use TF-IDF instead of word frequency');get_ipython().run_line_magic('pinfo', 'frequency')

💡 Why? TF-IDF normalizes term importance and reduces the weight of common terms.

✅ Solution:

python
Copy
Edit
vec = TfidfVectorizer()
dtm_tfidf = vec.fit_transform(sample['text'])


🔹 3. Remove Very Short Words (e.g., < 3 characters)
get_ipython().set_next_input('🧪 Question: How do you remove short meaningless words');get_ipython().run_line_magic('pinfo', 'words')

💡 Why? Short words like "an", "to", "go" are often semantically weak.

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(token_pattern=r'(?u)\b\w\w\w+\b')  # Words with 3 or more letters



🔹 4. Filter Rare Words (min_df)
get_ipython().set_next_input('🧪 Question: How do you ignore words that appear in only 1 or 2 tweets');get_ipython().run_line_magic('pinfo', 'tweets')

💡 Why? Rare words add noise and reduce model generalization.

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(min_df=2)



🔹 5. Filter Frequent Words (max_df)
get_ipython().set_next_input('🧪 Question: What if I want to ignore overly frequent words that appear in almost all documents');get_ipython().run_line_magic('pinfo', 'documents')

💡 Why? Extremely common words may not add value for classification.

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(max_df=0.90)  # Ignore words in more than 90% of tweets



🔹 6. Compare DTM With and Without Stopwords
get_ipython().set_next_input('🧪 Question: What’s the difference in feature count with and without stopwords');get_ipython().run_line_magic('pinfo', 'stopwords')

✅ Solution:

python
Copy
Edit
len(CountVectorizer().fit(sample['text']).get_feature_names_out())
len(CountVectorizer(stop_words='english').fit(sample['text']).get_feature_names_out())



🔹 7. Use Pre-tokenized Text (Avoid Double Tokenizing)
get_ipython().set_next_input('🧪 Question: What if the text is already tokenized');get_ipython().run_line_magic('pinfo', 'tokenized')

💡 Why? Applying CountVectorizer again would break it.

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)  # Pass-through
vec.fit_transform(sample['tokenized_column'])  # tokenized_column = list of tokens


🔹 8. Get Word Frequency Across All Tweets
get_ipython().set_next_input('🧪 Question: How do you find the most common words overall');get_ipython().run_line_magic('pinfo', 'overall')

✅ Solution:

python
Copy
Edit
dtm_df.sum(axis=0).sort_values(ascending=False).head(10)


🔹 9. Get Longest Tweet (by word count)
get_ipython().set_next_input('🧪 Question: How do you find which tweet is the longest');get_ipython().run_line_magic('pinfo', 'longest')

✅ Solution:

python
Copy
Edit
dtm_df.sum(axis=1).sort_values(ascending=False).head(1)


🔹 10. Visualize Word Frequencies
get_ipython().set_next_input('🧪 Question: Can you show the top 20 most frequent words in a bar chart');get_ipython().run_line_magic('pinfo', 'chart')

✅ Solution:

python
Copy
Edit
import matplotlib.pyplot as plt
dtm_df.sum(axis=0).sort_values(ascending=False).head(20).plot(kind='bar')
plt.title("Top 20 Words")
plt.show()
🔹 11. Fix Corrupt Characters
get_ipython().set_next_input('🧪 Question: How do you fix corrupted characters in text');get_ipython().run_line_magic('pinfo', 'text')

✅ Solution:

python
Copy
Edit
import ftfy
sample['text'] = sample['text'].apply(ftfy.fix_text)


🔹 12. Normalize Case (Lowercase All Words)
get_ipython().set_next_input('🧪 Question: How do you ensure ‘Game’ and ‘game’ are treated the same');get_ipython().run_line_magic('pinfo', 'same')

✅ Solution:

python
Copy
Edit
sample['text'] = sample['text'].str.lower()



🔹 13. Remove Numbers or Tokens with Digits
🧪 Question: How can you remove numeric tokens like ‘gme2023’?

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')



🔹 14. Extract N-grams (e.g., bigrams)
🧪 Question: Can you get phrase-level tokens like "short squeeze"?

✅ Solution:

python
Copy
Edit
vec = CountVectorizer(ngram_range=(2,2))  # For bigrams only



🔹 15. Get Vocabulary Size
get_ipython().set_next_input('🧪 Question: How many unique words are in your vectorizer');get_ipython().run_line_magic('pinfo', 'vectorizer')

✅ Solution:

python
Copy
Edit
len(vec.get_feature_names_out())



🔹 16. Export DTM for ML Use
get_ipython().set_next_input('🧪 Question: How would you export the document-term matrix for modeling');get_ipython().run_line_magic('pinfo', 'modeling')

✅ Solution:

python
Copy
Edit
dtm_df.to_csv('dtm_output.csv')


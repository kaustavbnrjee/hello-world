"""
@author: KB
BUSADM 797
Comprehensive Exam 2025

Sentiment Measure of Online Articles. GAMESTOP (GME) stock articles from Seeking Alpha
GME sample is the Seeking Alpha's sentiment of GME stock around 2018-2019 when the pumping happened
"""

import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
import os, pandas as pd, emoji, re, numpy as np
# pd.set_option('mode.chained_assignment',None)
# pd.set_option('display.max_columns',50)
# pd.set_option('display.width',200)

datafolder = "/Users/kaustav/Dropbox/BUSADM 797-01/Data"
stpath = f'{datafolder}/GMEstocktwitsV2_sample.csv.gz'
stdf = pd.read_csv(stpath)

sample = stdf.sample(1000,random_state=7) # get sample tweet, for testing code
test = sample['text'].tolist()[0]
Counter(word_tokenize(test))

# Create a new column which is a bag of words
sample['bow'] = sample['text'].apply(lambda x: Counter(word_tokenize(str(x))) if x is not None else Counter())
dtm1 = pd.DataFrame(sample['bow'].values.tolist()).fillna(0) # not the best way to generate
sample['text'] = sample['text'].fillna('').astype(str) # Replace np.nan with an empty string and convert all entries to string
vec = CountVectorizer()  # Initializes a word count vectorizer
dtm2 = vec.fit_transform(sample['text']) # Transforms text data into a sparse document-term matrix
dtm2.todense() # generates numpy matrix # np.asarray(dtm2.todense()) # generates numpy array

# example = pd.DataFrame(dtm2.todense(),
#              columns=vec.get_feature_names_out(),
#              index=sample.index)     # creates a df of wordcounts with proper labels for columns (words) and rows (documents).
"""
.todense() generates a matrix object, which behaves differently from a regular ndarray.
If you need the flexibility of working with a  general ndarray (which supports more operations and has better 
compatibility with most NumPy functions), you convert it using np.asarray()
"""
dtm2.sum(axis=1) # Total word counts by row that is per document.
dtm2.sum(axis=0) # Total counts of each word by column that is across all documents.

example = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)     #Creates a DataFrame of word counts with proper labels for columns (words) and rows (documents).


example.sum(axis=1) # Total word counts by row that is per document.
example.sum(axis=0) # Total counts of each word by column that is across all documents.
example_1_df = pd.DataFrame(example.sum(axis=1))
example_1_df.to_csv("example_1.csv")

example_2_df = pd.DataFrame(example.sum(axis=0))
example_2_df.to_csv("example_2.csv")

"""
Another way to do this is using the ftfy library. This is used here to fix encoding errors and clean up of text data.
Specifically, ftfy.ftfy is a function that attempts to "fix" common problems with text, such as garbled characters, 
invalid Unicode sequences, or other encoding issues.
"""
import ftfy 

# a = ftfy.ftfy('ªð') # an example
sample['text'] = sample['text'].apply(ftfy.ftfy)    # cleans text data by fixing encoding issues using the ftfy library
dtm2 = vec.fit_transform(sample['text'])    # converts the cleaned text into a Document-Term-Matrix of word counts using CountVectorizer
dtm2_df = pd.DataFrame(dtm2.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)

#Converts the sparse matrix of word counts into a dense matrix and 
#creates a DataFrame, labeling the columns with the words (features)
#and keeping the original document indices.

dtm2_df.sum(axis=0).sort_values(ascending=False)    # Total word counts by row that is per document.
dtm2_df.sum(axis=1).sort_values(ascending=False)    # Total counts of each word by column that is across all documents.

# Create another dataset without stopwords
from nltk.corpus import stopwords
stops = stopwords.words('english')   
vec = CountVectorizer(stop_words=stops,min_df=2)
#This argument passes the stops list (which contains the English stopwords) to the CountVectorizer.
# As a result, any word in the stopwords list will be ignored or filtered out when the CountVectorizer processes the text

# min_df=2 # only words that appear in at least 2 documents will be considered
dtm2b = vec.fit_transform(sample['text'])
dtm2b_df = pd.DataFrame(dtm2b.todense(),
             columns=vec.get_feature_names_out(),
             index=sample.index)
dtm2b_df.sum(axis=0).sort_values(ascending=False)   # Total word counts by row that is per document.
dtm2b_df.sum(axis=1).sort_values(ascending=False)   # Total counts of each word by column that is across all documents.
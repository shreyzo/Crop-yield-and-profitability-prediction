# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:35:29 2020

@author: user
"""

# Define the documents
doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"

doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"

doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"

documents = [doc_trump, doc_election, doc_putin]

# Imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

## Using CountVectorizer ##
# Creating the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)
# Converting Sparse Matrix to Pandas Dataframe to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
              	columns=count_vectorizer.get_feature_names(),
              	index=['doc_trump', 'doc_election', 'doc_putin'])
print("Document Term Matrix using CountVectorizer:")
print(df)
print()
# Computing Cosine Similarity
print("Cosine Similarity using CountVectorizer:")
print(cosine_similarity(df, df))

print()
print()

## Using TfidfVectorizer ##
# Creating the Document Term Matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer()
sparse_matrix = tfidf_vectorizer.fit_transform(documents)
# Converting Sparse Matrix to Pandas Dataframe to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
              	columns=tfidf_vectorizer.get_feature_names(),
              	index=['doc_trump', 'doc_election', 'doc_putin'])
print("Document Term Matrix using TfidfVectorizer:")
print(df)
print()
# Computing Cosine Similarity
print("Cosine Similarity of TfidfVectorizer:")
print(cosine_similarity(df, df))

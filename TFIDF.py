phrases=["le chat mange des croquettes","le chien aime ses croquettes le chien le","le chat ronronne et mange"]
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer=CountVectorizer()
vector=vectorizer.fit_transform(phrases)
import pandas as pd
import numpy as np

pd.DataFrame(vector.toarray(),columns=vectorizer.get_feature_names())

# ****************************************

tfidf_vectorizer=TfidfVectorizer()
tfidfvector=tfidf_vectorizer.fit_transform(phrases)
pd.DataFrame(np.around(tfidfvector.toarray(),2),columns= tfidf_vectorizer.get_feature_names())

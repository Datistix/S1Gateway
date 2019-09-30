##### Import Libraries #####

# manipulate data
import pandas as pd
import numpy as np
import unicodedata
from stop_words import get_stop_words

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# cross-validation
from sklearn.model_selection import train_test_split


# language manipualtion
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SpanishStemmer

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# metrics
from sklearn.metrics import confusion_matrix

##### Load Data #####

df_0=pd.read_excel('frases_modelo.xlsx')
stop_words_0 = get_stop_words('es')

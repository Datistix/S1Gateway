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

##### Post Loading #####

# avoid load raw data from path
df = df_0[:]
stop_words = stop_words_0

##### To Lower Case #####

# to lower case columns
df.columns = ['intent', 'phrasenlp', 'phrase']

# drop phrasenlp column
df=df[['intent', 'phrase']]

# phrase to str
df['phrase'] = df['phrase'].astype(str) 

##### Normalize #####

# normalize function. lower case, errase certian characters
def normalize_title(title):
    return unicodedata.normalize('NFKD', title.lower()).encode('ASCII', 'ignore').decode('utf8')

# apply normalize function
df['phrase'] = df.phrase.apply(normalize_title)

##### Exploratory Data Analysis (EDA) ######

# top 5 dataset info
df.head()

categories = len(df.intent.unique())
print ("There are ", categories,  " categories.")

questions = len(df)
print ("There are ", questions,  " questions.")

df.groupby(['intent']).count().sort_values(by='phrase' ,ascending=False).head(10)

df.groupby(['intent']).count().sort_values(by='phrase' ,ascending=True).head(10)

##### Visualization #####

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white",
                      stopwords = (["quiero", "puedo", "saber", 'cuanto'] + list(stop_words))).generate(text = " ".join(review for review in df.phrase))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.figure(figsize=(10,4))
fig = sns.countplot(y = 'intent',data=df, palette = sns.cubehelix_palette(10, start=2.95, rot=0),
                   order=pd.value_counts(df['intent']).iloc[:10].index)
fig.set_title('Top 10 Categories', pad=25)
fig.set_xlabel('Observations', labelpad=25)
fig.set_ylabel('Categories', labelpad=25)


plt.show()

##### NA Values #####

df.dropna(inplace=True)

len(df)

# no NA Values

##### Outliers #####

# drop data contains
intent = df.groupby('intent', as_index = False).count()
intent = np.array(intent[intent.phrase>20].intent)
df = df[df.intent.isin(intent)]


##### CountVectorizer #####

# stemming
stemmer = SpanishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

# define vectorizer parameters
vectorizer = CountVectorizer(encoding='utf-8', strip_accents='ascii', lowercase=False, stop_words = stop_words,
                            analyzer=stemmed_words)

# apply vectorizer
df_vec = vectorizer.fit_transform(df.phrase)

tokens = len(vectorizer.get_feature_names())
print ("There are ", tokens,  " tokens.")

vectorizer.vocabulary_

##### Train Test Split ######

x_train,x_test,y_train,y_test = train_test_split (df_vec, df.intent, test_size=0.3, random_state=1000)
classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

#pip install wordcloud
# from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv('review_shopping.csv', sep='\t', names=['text', 'sentiment'], header=None)
# print(df)

show_bar = df['sentiment'].value_counts().plot.bar()
# print(show_bar)

thai_stopwords = list(thai_stopwords())
print(thai_stopwords)

def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split()
                     if word.lower not in thai_stopwords)
    return final
df['text_tokens'] = df['text'].apply(text_process)
# print(df)


X = df[['text_tokens']]
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
cvec.fit_transform(X_train['text_tokens'])
# print(cvec.vocabulary_)

train_bow = cvec.transform(X_train['text_tokens'])
pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names_out(), index=X_train['text_tokens'])

lr = LogisticRegression()
lr.fit(train_bow, y_train)

test_bow = cvec.transform(X_test['text_tokens'])
test_predictions = lr.predict(test_bow)
# print(classification_report(test_predictions, y_test))

#test your message
my_text = 'คุ้ม'
my_tokens = text_process(my_text)
my_bow = cvec.transform(pd.Series([my_tokens]))
my_predictions = lr.predict(my_bow)
print(my_predictions)


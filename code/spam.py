# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

from nltk.corpus import stopwords

# tokenization and pre-processing
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# count vector
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


message4 = messages['message'][3]

bow4 = bow_transformer.transform([message4])

messages_bow = bow_transformer.transform(messages['message'])

#sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
#print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
#print(tfidf4)

messages_tfidf = tfidf_transformer.transform(messages_bow)
#print(messages_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

all_predictions = spam_detect_model.predict(messages_tfidf)
#print(all_predictions)


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

def ask():
    arr = list()
    input_string = str(input('submit comment = '))
    arr.append(input_string)
    cou = bow_transformer.transform(arr)
    pred_orig = spam_detect_model.predict(cou)
    print(pred_orig[0])

ask()































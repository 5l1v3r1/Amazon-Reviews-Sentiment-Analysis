import math
from ctypes import c_char_p

import pandas as pd
import re
import multiprocessing
from nltk.tokenize import word_tokenize as wt

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset = pd.read_csv('dataset/foods.csv', sep=";", quotechar="|", encoding='ISO-8859-1')

manager = multiprocessing.Manager()
data = manager.list()

stop_words = stopwords.words('english')


def cleanup_data(data_chunk):
    print "Cleanup started."
    for i in range(data_chunk.shape[0]):
        text = data_chunk.iloc[i, 6]

        # remove non alphabatic characters
        text = re.sub("[^A-Za-z]", ' ', text)

        # make words lowercase, because Go and go will be considered as two words
        text = text.lower()

        # tokenising
        tokenizing_text = wt(text)

        # remove stop words
        text_processed = [w for w in tokenizing_text if w not in stop_words]

        tmp_text = " ".join(text_processed)
        global data
        data.append(tmp_text)
        if i % 10000 == 0:
            print multiprocessing.current_process(), " ", i


processes = []
nprocs = 8
chunksize = int(math.ceil(dataset.shape[0] / float(nprocs)))
for p in range(nprocs):
    chunk = dataset.iloc[(chunksize * p): (chunksize * (p + 1))]
    process = multiprocessing.Process(target=cleanup_data, args=(chunk,))
    processes.append(process)
    process.start()

for process in processes:
    process.join()

print len(data)

# creating the feature matrix
matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(data).toarray()
dataset.ix[:, 3] = dataset.ix[:, 3].apply(pd.to_numeric)
y = dataset.iloc[:, 3]

# split train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict class
y_pred = classifier.predict(X_test)

# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
print cm
cr = classification_report(y_test, y_pred)
print cr

accuracy = accuracy_score(y_test, y_pred)
print accuracy

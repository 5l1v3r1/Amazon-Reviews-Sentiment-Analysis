import csv
import math
import re
import multiprocessing

import pandas as pd
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords

dataset = pd.read_csv('dataset/foods.csv', sep=";", quotechar="|", encoding='ISO-8859-1')

manager = multiprocessing.Manager()
data = manager.list()
print(type(data))

stop_words = stopwords.words('english')


def cleanup_data(data_chunk):
    print("Cleanup started.")
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
            print(multiprocessing.current_process(), " ", i)


# clean the data with 8 processes
def cleanup_data_mp():
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


def generate_clean_file(chunk, begin, end):
    print("Started", multiprocessing.current_process())
    chunk_copy = chunk.copy()
    for i in range(begin, end):
        chunk_copy.loc[i, 6] = data[i]
    pd.DataFrame(chunk_copy).to_csv("/Users/boran/Documents/CS464_Project/dataset/foods_clean%s.csv" % begin, sep=";", quotechar="|", quoting=csv.QUOTE_NONNUMERIC)
    print("generated ", multiprocessing.current_process())


def generate_clean_file_mp():
    processes = []
    nprocs = 8
    chunksize = int(math.ceil(dataset.shape[0] / float(nprocs)))
    for p in range(nprocs):
        chunk = dataset.iloc[(chunksize * p): (chunksize * (p + 1))]
        process = multiprocessing.Process(target=generate_clean_file,
                                          args=(chunk, (chunksize * p), (chunksize * (p + 1))))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

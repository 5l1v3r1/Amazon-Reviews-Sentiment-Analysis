import csv
from random import shuffle

'''
Don't need to modify this file again, it has generated the necessary files already. 
Uploading just for backup.
'''

with open('foods.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_ALL)

    filepath = 'foods.txt'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            # [:-1] is for removing \n from string
            product_id = line.split(' ')[1][:-1]
            user_id = fp.readline().split(' ')[1][:-1]
            fp.readline()
            helpfulness = fp.readline().split(' ')[1][:-1]
            score = fp.readline().split(' ')[1][:-1]
            time = fp.readline().split(' ')[1][:-1]
            summary = fp.readline().split('summary: ')[1][:-1]
            text = fp.readline().split('text: ')[1][:-1]

            fp.readline()
            line = fp.readline()

            filewriter.writerow([product_id, user_id, helpfulness, score, time, summary, text])


def shuffle_list(rlist):
    shuffle(rlist)


def fill_train_and_test_list(review_list):
    split_point = int(0.8 * len(review_list))
    train_list = review_list[:split_point]
    test_list = review_list[split_point:]

    return train_list, test_list


def write_list_to_csv(filename, rlist):
    with open(filename, 'wb') as traing_file:
        writer = csv.writer(traing_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_ALL)
        for review in rlist:
            writer.writerow([review.product_id, review.user_id, review.helpfulness, review.score, review.time,
                                 review.summary, review.text])
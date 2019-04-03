from random import shuffle

from Review import Review

review_list = []

train_list = []
test_list = []


def read_file():
    filepath = '/Users/boranyildirim/Documents/CS464Project/foods.txt'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            product_id = line.split(' ')[1]
            user_id = fp.readline().split(' ')[1]
            fp.readline()
            helpfulness = fp.readline().split(' ')[1]
            score = fp.readline().split(' ')[1]
            time = fp.readline().split(' ')[1]
            summary = fp.readline().split('summary: ')[1]
            text = fp.readline().split('text: ')[1]

            review = Review(product_id, user_id, helpfulness, score, time, summary, text)
            review_list.append(review)

            fp.readline()
            line = fp.readline()
            cnt += 1
            if cnt % 10000 == 0:
                print cnt


def shuffle_review_list():
    shuffle(review_list)


def fill_train_and_test_list():
    split_point = int(0.8 * len(review_list))
    global train_list
    train_list = review_list[:split_point, :]
    global test_list
    test_list = review_list[split_point:, :]

    global review_list
    del review_list


if __name__ == '__main__':
    read_file()
    shuffle_review_list()
    fill_train_and_test_list()
    print len(train_list)
    print len(test_list)

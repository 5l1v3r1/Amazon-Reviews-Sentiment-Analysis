import csv
from Review import Review

train_list = []
test_list = []


def read_file(filename, rlist):
    with open(filename, 'rb') as foods_file:
        reader = csv.reader(foods_file, delimiter=';', quotechar="|")

        # read file row by row
        for row in reader:
            review = Review(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
            rlist.append(review)


if __name__ == '__main__':
    read_file("dataset/train.csv", train_list)
    read_file("dataset/test.csv", test_list)

    print len(train_list)
    print len(test_list)

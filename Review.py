from nltk import word_tokenize
from nltk.corpus import stopwords


class Review:

    def __init__(self, product_id, user_id, helpfulness, score, time, summary, text):
        self.product_id = product_id
        self.user_id = user_id
        self.helpfulness = helpfulness
        self.score = score
        self.time = time
        self.summary = summary
        self.text = summary

    ''' don't need to call remove_stop_words because test and train files are already generated
    Uploading just for backup.'''
    @staticmethod
    def remove_stop_words(text):
        # list of all stop words
        stop_words = set(stopwords.words('english'))
        # tokenize the text
        word_tokens = word_tokenize(text)
        # make all tokens lowercase
        tokens = [token.lower() for token in word_tokens]
        # remove stopwords from text
        filtered_text = [w for w in tokens if w not in stop_words]

        filtered_text_with_counts = {w: filtered_text.count(w) for w in filtered_text}

        return filtered_text_with_counts

    def __str__(self):
        return str(self.__dict__)

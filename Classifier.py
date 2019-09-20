import glob  # to read all files from the folder
from nltk.tokenize import word_tokenize
import io  # to encode files
from sklearn.metrics import accuracy_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
import pickle


class Classifier:
    def __init__(self, negdir, posdir, txt):
        self.negdir = negdir
        self.posdir = posdir
        self.txt = txt

    def read_file(self):
        reviews = []
        labels = []  # 1 is pos, 0 is neg

        negatives = glob.glob(self.negdir + '/*.txt')

        for file in negatives:
            with io.open(file, 'rt', encoding='utf-8') as f:
                review = f.read().lower()
                reviews.append(review)
                labels.append(0)

        positives = glob.glob(self.posdir + '/*.txt')

        for file in positives:
            with io.open(file, 'rt', encoding='utf-8') as f:
                review = f.read().lower()
                reviews.append(review)
                labels.append(1)

        with open('rev_lab.pickle', 'wb') as f:
            pickle.dump(reviews, f)
            pickle.dump(labels, f)

    def fit(self):

        with open('rev_lab.pickle', 'rb') as f:
            reviews = pickle.load(f)
            labels = pickle.load(f)

        vectorizer = CountVectorizer(min_df=2, tokenizer=word_tokenize)
        counts = vectorizer.fit_transform(reviews)  # fitted with train data

        transformer = TfidfTransformer()
        tf_idf = transformer.fit_transform(counts)  # the same as vectorizer

        X_train, X_test, y_train, y_test = train_test_split(tf_idf, labels, train_size=0.9, random_state=42)

        classifier = LogisticRegression(C=7, solver='liblinear')
        classifier.fit(X_train, y_train)

        print(accuracy_score(y_test, classifier.predict(X_test)))
        print(recall_score(y_test, classifier.predict(X_test)))

        with open('regression.pickle', 'wb') as f:
            pickle.dump(classifier, f)
            pickle.dump(vectorizer, f)
            pickle.dump(transformer, f)

    def classify(self):

        with open('regression.pickle', 'rb') as f:
            classifier = pickle.load(f)
            vectorizer = pickle.load(f)
            transformer = pickle.load(f)

        txt_counts = vectorizer.transform(self.txt)  # only transformation is needed for input
        tf_idf_txt = transformer.transform(txt_counts)

        sentiment = classifier.predict(tf_idf_txt)
        print('sentiment is ' + str(sentiment))


C = Classifier('aclImdb/train/neg', 'aclImdb/train/pos', ["Blake Edwards' legendary fiasco, begins to seem pointless after just 10 minutes. A combination of The Eagle Has Landed, Star!, Oh! What a Lovely War!, and Edwards' Pink Panther films, Darling Lili never engages the viewer; the aerial sequences, the musical numbers, the romance, the comedy, and the espionage are all ho hum. At what point is the viewer supposed to give a damn? This disaster wavers in tone, never decides what it wants to be, and apparently thinks it's a spoof, but it's pathetically and grindingly square. Old fashioned in the worst sense, audiences understandably stayed away in droves. It's awful. James Garner would have been a vast improvement over Hudson who is just cardboard, and he doesn't connect with Andrews and vice versa. And both Andrews and Hudson don't seem to have been let in on the joke and perform with a miscalculated earnestness. Blake Edwards' SOB isn't much more than OK, but it's the only good that ever came out of Darling Lili. The expensive and professional look of much of Darling Lili, only make what it's all lavished on even more difficult to bear. To quote Paramount chief Robert Evans, '24 million dollars worth of film and no picture'"])
C.fit()


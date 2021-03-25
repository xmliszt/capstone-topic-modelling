from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_punctuation2, preprocess_string, strip_short, strip_numeric, stem_text, strip_tags
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
import logging
import warnings
from gensim import corpora, models, similarities
import gensim
from collections import defaultdict
import os
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

logging.basicConfig(filename='classifier.log',
                    level=logging.INFO)

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]


def id2categories() -> dict:
    o = dict()
    for idx, cat in enumerate(CATEGORIES):
        o[idx] = cat
    return o


def categories2id() -> dict:
    o = dict()
    for idx, cat in enumerate(CATEGORIES):
        o[cat] = idx
    return o


# Define custom filters
CUSTOM_FILTERS = [lambda x:
                  x.lower(),
                  strip_multiple_whitespaces,
                  strip_numeric,
                  remove_stopwords,
                  strip_short,
                  stem_text,
                  strip_tags,
                  strip_punctuation2
                  ]


STOP_WORDS = stopwords.words('english')


def load_stopwords() -> list:
    stop_words = []
    with open("stopword", "r") as fh:
        words = fh.readlines()
        for word in words:
            word = word.rstrip("\n").strip()
            stop_words.append(word)
    return stop_words


STOP_WORDS.extend(load_stopwords())


DATA_PATH = "./BBC News Summary/BBC News Summary/News Articles"

# Load the text from directory, identify category from folder and
# put the texts in a list in the order of the categories specified in configurables


def get_texts() -> (list, list, list):
    category_text_map = defaultdict(list)
    texts = []
    categories = []
    ignored = []
    ignore = False
    for root, _, files in os.walk(DATA_PATH):
        if len(files) == 0:
            continue
        category = root.split('/')[-1].lower()
        if category not in CATEGORIES:
            print("Category {} is not in pre-set categories. Please add it in and re-run the program!".format(category))
            ignore = True
        else:
            ignore = False
        for f in files:
            txt_path = os.path.join(root, f)
            text = ""
            with open(txt_path, 'r', encoding="ISO-8859-1") as fh:
                lines = fh.readlines()
                for line in lines:
                    text += line
            if ignore:
                ignored.append(text)
            else:
                category_text_map[category].append(text)

    for cat in CATEGORIES:
        files = category_text_map[cat]
        texts.extend(files)
        categories.extend([categories2id()[cat] for _ in range(len(files))])

    if len(texts) != len(categories):
        raise Exception("Number of articles and number of target categories do not have the same length: [{} != {}]".format(
            len(texts), len(categories)))

    return texts, categories, ignored


def clean_texts(texts: list) -> list:
    clean_texts = []
    for text in texts:
        processed_texts = preprocess_string(text, CUSTOM_FILTERS)
        processed_texts = [w for w in processed_texts if not w in STOP_WORDS]
        clean_texts.append(processed_texts)
    return clean_texts


texts, categories, ignored = get_texts()
texts = clean_texts(texts)
logging.info(
    "{} articles loaded. {} articles ignored due to non-existing categories.".format(len(texts), len(ignored)))


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_corpus(texts, bigram_mod):
    bigram_mod = bigrams(texts)
    bigram = [bigram_mod[text] for text in texts]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


bigram_mod = bigrams(texts)
corpus, dictionary, bigram = get_corpus(texts, bigram_mod)


def create_lda_model(corpus, dictionary, num_topics):
    if not os.path.exists("models"):
        os.mkdir("models")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda_train = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            chunksize=100,
            workers=7,  # Num. Processing Cores - 1
            passes=50,
            eval_every=1,
            per_word_topics=True)
        lda_train.save(os.path.join("models", "lda_train.model"))
    return lda_train


train_lda_model = create_lda_model(corpus, dictionary, len(CATEGORIES))
logging.info(train_lda_model)


def get_feature_vectors(texts: list, corpus, lda_model):
    train_vecs = []
    for i in range(len(texts)):
        top_topics = lda_model.get_document_topics(
            corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(len(CATEGORIES))]
        topic_vec.append(len(texts))
        topic_vec.append(len(texts[i]))
        train_vecs.append(topic_vec)
    return train_vecs


# Training classifier

texts_corpus_zip = list(zip(texts, corpus))

train_texts, test_texts, train_categories, test_categories = train_test_split(
    texts, categories, test_size=0.2, shuffle=True)

feature_vectors = get_feature_vectors(train_texts, corpus, train_lda_model)
logging.info(len(feature_vectors), len(train_categories))


def create_model(activation="relu", optimizer="adam", dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(32, input_dim=len(CATEGORIES)+2, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, input_dim=len(CATEGORIES)+2, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, input_dim=len(CATEGORIES)+2, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, input_dim=len(CATEGORIES)+2, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(CATEGORIES), activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model


OPTIMIZERS = ['SGD', 'RMSprop', 'Adagrad',
              'Adadelta', 'Adam', 'Adamax', 'Nadam']
ACTIVATION = ['relu', 'tanh', 'sigmoid', 'linear']
BATCH_SIZE = [5, 10, 20, 30, 40, 50]
EPOCHS = [10, 50, 100]
DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

param_grid = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": OPTIMIZERS,
    "dropout_rate": DROPOUT,
    "activation": ACTIVATION,
}


def train(train_vecs, targets):
    X = np.array(train_vecs)
    Y = np_utils.to_categorical(np.array(targets))

    model = KerasClassifier(build_fn=create_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        n_jobs=-1, cv=10, verbose=5)

    grid_result = grid.fit(X, Y)
    logging.info("Best: %f using %s" %
                 (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        logging.info("%f (%f) with: %r" % (mean, stdev, param))

    best_model = grid_result.best_estimator_
    best_model.save(os.path.join("models", "classifier.pth"))


train(feature_vectors, train_categories)

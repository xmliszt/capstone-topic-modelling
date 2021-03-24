from pprint import pprint
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import preprocess_string
from src.custom_filter import CUSTOM_FILTERS
from src.tweet import get_user_tweets, connect_to_twitter_OAuth, get_search_tweets
from src.reader import get_article
import pyLDAvis.gensim
import sys

# Get texts
# api = connect_to_twitter_OAuth()
# texts = get_search_tweets(api, "Trump")
texts = get_article(sys.argv[1])

# Custom Pre-processing with Filters
idx = 0
while idx < len(texts):
    texts[idx] = preprocess_string(texts[idx], CUSTOM_FILTERS)
    idx += 1

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# print([[(dictionary[id], count) for id, count in v] for v in corpus[:1]])

# Document similarity - Vector
# 1. How many times does the word splonge appear in the document? Zero.
# 2. How many paragraphs does the document consist of? Two.
# 3. How many fonts does the document use? Five.
# (2, 2.0), (3, 5.0)
# Compare vector to find similarity

# Token - ID pairs
# pprint(dictionary.token2id)

# TFIDF transformation
tfidf = models.TfidfModel(corpus)
transformed_tfidf = tfidf[corpus]
lda = models.LdaMulticore(
    transformed_tfidf, num_topics=5, id2word=dictionary)
# print(lda.show_topics())

# Visualize topics
data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
with open('index.html', 'w') as fp:
    pyLDAvis.save_html(data, fp)

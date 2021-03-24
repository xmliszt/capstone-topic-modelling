import tweepy

# Variables that contains the credentials to access Twitter API
ACCESS_TOKEN = ''
ACCESS_SECRET = ''
CONSUMER_KEY = ''
CONSUMER_SECRET = ''


# Setup access to API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth)
    return api


def get_user_tweets(api, user="BarackObama"):
    texts = []
    tweets = api.user_timeline(user)
    for tweet in tweets:
        texts.append(tweet.text)
    return texts


def get_search_tweets(api, q):
    texts = []
    results = api.search(q, lang="en")
    for r in results:
        texts.append(r.text)
    return texts

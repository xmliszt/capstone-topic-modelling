import json
import os


def get_article(path):
    article = []
    if os.path.basename(path).split('.')[1].lower() != "json":
        raise TypeError("File must be in JSON format!")
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        body = data['body']
        for _, v in body.items():
            article.append(v)
    return article

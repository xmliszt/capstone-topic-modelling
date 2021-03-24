from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_punctuation2, preprocess_string, strip_short, strip_numeric, stem_text, strip_tags

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

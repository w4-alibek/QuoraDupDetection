#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

def clean_text(text):
    text = text\
        .lower()\
        .replace(",000,000", "m")\
        .replace(",000", "k")\
        .replace("′", "'")\
        .replace("’", "'")\
        .replace("won't", "will not")\
        .replace("cannot", "can not")\
        .replace("can't", "can not")\
        .replace("n't", " not")\
        .replace("what's", "what is")\
        .replace("it's", "it is")\
        .replace("'ve", " have")\
        .replace("i'm", "i am")\
        .replace("'re", " are")\
        .replace("he's", "he is")\
        .replace("she's", "she is")\
        .replace("'s", " own")\
        .replace("%", " percent ")\
        .replace("₹", " rupee ")\
        .replace("$", " dollar ")\
        .replace("€", " euro ")\
        .replace("'ll", " will")\
        .replace("=", " equal ")\
        .replace("+", " plus ")

    text = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', text)
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    return text
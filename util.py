"""Utility functions
"""
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import string


def remove_stop_words_and_punctuation(text):
    """Given text as string. Removes stop words from text and
    return list of words without stop word
    """
    stop_words = stopwords.words('english')
    stop_words.extend(list(string.punctuation))
    stop_words = set(stop_words)
    word_tokens = word_tokenize(text.decode('utf-8'))
    filtered_words = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_words)


def normalize_words(words):
    """Normalize the given list of words. Return list of normalized word
    """
    stemmer = SnowballStemmer("english")
    for index, word in enumerate(words):
        words[index] = stemmer.stem(word)

def subliner_term_frequency(word, tokenized_words):
    """Given sentence as list of the words.
    Return sublinear word freq
    """
    count = tokenized_words.count(word)
    if count == 0:
        return 0
    return 1 + math.log(count)

def inverse_document_frequencies(tokenized_sentences):
    """tokenized_sentences: list [[w00...wN0],...,[w0...wNK]] and sentences as list of the word.
    Return list freqs of the sentence with the specific word
    """
    idf_values = {}
    set_of_words = set([item for sublist in tokenized_sentences for item in sublist])
    for word in set_of_words:
        contains_token = 0
        for sentence in tokenized_sentences:
            contains_token += word in sentence
        idf_values[word] = 1 + math.log(len(tokenized_sentences)/contains_token)
    return idf_values

def tfxidf(sentences):
    """Given the [[w00...wN0],...,[w0...wNK]] and sentences as list of the word.
    Return [[w00f...wM0f],...,[w0Kf...wMKf]] every wmkf means word freq for sentence
    """
    idf = inverse_document_frequencies(sentences)
    tfidf_sentences = []
    for sentence in sentences:
        sentence_tfidf = []
        for word in idf.keys():
            tf = subliner_term_frequency(word, sentence)
            sentence_tfidf.append(tf * idf[word])
        tfidf_sentences.append(sentence_tfidf)
    return tfidf_sentences

def main():
    """How to use util functions
    """
    #list type
    document_0 = remove_stop_words("China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy.")
    document_1 = remove_stop_words("At last, China seems serious about confronting an endemic problem: domestic violence and corruption.")
    document_2 = remove_stop_words("Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people.")
    document_3 = remove_stop_words("Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled.")
    document_4 = remove_stop_words("What's the future of Abenomics? We asked Shinzo Abe for his views")
    document_5 = remove_stop_words("Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily.")
    document_6 = remove_stop_words("Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?")

    normalize_words(document_0)
    normalize_words(document_1)
    normalize_words(document_2)
    normalize_words(document_3)
    normalize_words(document_4)
    normalize_words(document_5)
    normalize_words(document_6)

    documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

    TFIDF = tfxidf(documents)

if __name__ == "__main__":
    main()

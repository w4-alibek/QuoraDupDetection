"""How to use:

## find all words
words = {}
read_quora_words(words, 'train.csv')
read_glove_words(words, glove_filename)

## build index
word_to_id = build_word_to_id(words)

## for later use
save_word_to_id(word_to_id, 'word_to_id.csv')

## map tokens in question pairs to their word id
quora_raw_data('train.csv', word_to_id)

"""

import datasets

def read_glove_words(words, filename):
    """Add words from glove to the dictionary
    """
    for line in datasets.glove_dataset(0, filename):
        if line[0]:
            words[line[0]] = None

def read_quora_words(words, filename):
    """Find all words in quora corpus
    """
    question_index_1 = 3
    question_index_2 = 4

    for row in datasets.quora_dataset(filename):
        qwords = row[question_index_1].split()
        qwords.extend(row[question_index_2].split())

        for word in qwords:
            if word:
                words[word] = None


def build_word_to_id(words):
    """Build word_to_id given all words
    """
    return dict(zip(words, range(len(words))))


def build_tokenized_samples(data_path, word_to_id, remove_stopwords):
    """Tokenize each sample into word ids
    """
    samples = []
    for row in datasets.quora_dataset(remove_stopwords, data_path):
        tokens1 = map(word_to_id.get, row[3])
        tokens2 = map(word_to_id.get, row[4])
        sample = {
            'q1': {
                'id': row[1],
                'tokens': [x if x else word_to_id['abc'] for x in tokens1]
            },
            'q2': {
                'id': row[2],
                'tokens': [x if x else word_to_id['abc'] for x in tokens2]
            },
            'label': row[5]
        }
        samples.append(sample)

    return samples

def save_samples(samples):
    """Save to disk
    """
    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in samples.items():
            writer.writerow([key, value])
